# 把训练切换到 Muon(HybridMuon)优化器 —— 移植与对齐要点

> 目标读者:负责 **eSEN** 项目的 Claude Code。
> 目的:在 eSEN 里复现本项目(equiformer_v3)的 **HybridMuon** 优化器,并与本项目对齐。
> 本项目权威实现:`src/fairchem/core/common/muon.py`(直接读它做参考)。
> 接线参考:`src/fairchem/core/trainers/base_trainer.py`(搜 `HybridMuon` / `build_hybrid_muon_param_groups`)。

---

## 0. TL;DR(最重要的 5 条)

1. **HybridMuon = Muon(管所有 ≥2D 权重矩阵) + AdamW(管其余:bias / norm / embedding)**。不是纯 Muon。
2. **`update_scale` 选 `moonlight`**(`0.2*sqrt(max(rows,cols))`),不要用 `ratio`。moonlight 让同一个 `muon_lr` 跨模型尺寸可迁移;ratio 的有效 lr 依赖矩阵形状,换模型必须重新调。
3. **muon_lr 与 AdamW lr 是两套**:`lr_initial`(yml)= 非矩阵参数的 AdamW lr;`optimizer_params.muon_lr` = 矩阵的 Muon lr。scheduler 同时缩放两者。
4. **发散保护里的 spike 守卫当前有死锁 bug,移植时必须修**(见 §6)。不修的话:大模型/长训练会出现"跳步数单调暴涨→大半权重被永久冻结→力/精度学不动但不报错"的隐性炸裂。
5. **梯度力/二阶反向 + Wigner 等 fp32 敏感路径不要开 AMP**(equiformer 特有,见 §5;eSEN 按自己的精度约束对照)。

---

## 1. 核心设计:HybridMuon

Muon(Newton-Schulz 正交化动量)只对**矩阵**有定义,所以非矩阵参数交给 AdamW。一个 `torch.optim.Optimizer` 子类内部同时跑两条 path,按 param-group 上的 `use_muon` 布尔标志分流:

- `use_muon=True` → Muon 更新,用 `lr, momentum, nesterov, ns_steps, weight_decay`。
- `use_muon=False` → 解耦 AdamW 更新,用 `lr, betas, eps, weight_decay`。

## 2. 参数分组规则(关键,容易抄错)

见 `build_hybrid_muon_param_groups`。规则:

| 参数 | 去向 | weight_decay |
|---|---|---|
| 名字命中 `no_weight_decay` 集合(bias / norm 增益 / embedding) | AdamW | **0** |
| 其余且 `ndim >= 2`(Linear / 任意张量权重) | **Muon** | wd |
| 其余 1D 漏网的 | AdamW | wd |

注意点:
- 用 **`name.endswith(suffix)`** 匹配,而不是精确等于 —— 这样 DDP/compile 包出来的 `module.` / `_orig_mod.` 前缀仍能命中。eSEN 若有自己的 `no_weight_decay` 约定,直接复用其语义。
- 判据是 `p.ndim >= 2`,所以 **3D/4D 权重也会进 Muon**(本项目 `SO3Linear` 是 `(lmax+1, out, in)` 3D)。eSEN 里要确认哪些是"真矩阵权重",哪些是不该正交化的张量(见 §4)。

## 3. `update_scale`:ratio vs moonlight(决定 lr 可迁移性)

正交化后 NS 输出 ~半正交,逐元素 RMS ≈ `1/sqrt(max(rows,cols))`。两种缩放把它变成实际步长:

- `ratio`(Keller-Jordan):`scale = max(1, rows/cols)**0.5` → 更新 RMS ≈ `lr/sqrt(fan_in)`,**形状相关** → 换 C / ffn_hidden / 深度后 muon_lr 必须重调。
- `moonlight`(Moonshot,arXiv:2502.16982):`scale = 0.2*sqrt(max(rows,cols))` → 更新 RMS ≈ `0.2*lr` 对**每个矩阵都一样**,与形状无关 → **muon_lr 标定一次即可跨尺寸复用**。

**推荐 `moonlight`。** 两种模式量纲不同,切换模式必须重标 muon_lr(ratio 的 lr 拿到 moonlight 没意义,反之亦然)。

## 4. Newton-Schulz 正交化(注意张量维度)

`zeropower_via_newtonschulz5`:
- 只正交化**最后两维**,对前面的 leading dims 做 batch。所以 `(out,in)` 和 `(lmax+1,out,in)` 都能直接处理。
- 内部转 **bfloat16** 跑迭代(速度/稳定),再转回原 dtype。
- 先转成"宽"朝向(让 `X@X^T` 取小的那个),再按谱范数归一。
- `ns_steps=5` 是默认迭代步数。

**eSEN 移植注意**:确认进 Muon 的每个权重"最后两维确实是要正交化的矩阵"。卷积核 `(out,in,kh,kw)` 这类把最后两维当成 kernel 空间维直接正交化是**错的**;要么 reshape 成 `(out, in*kh*kw)` 再走 Muon,要么把它划给 AdamW。eSEN 是否有这类层,移植前先扫一遍。

## 5. 分布式 / AMP / scheduler 兼容

- **DDP**:每个 rank 梯度先 all-reduce(平均)再 `step()`,所以各 rank 算出**完全相同**的 Muon 更新,副本天然同步。**纯 DDP 不需要 ZeRO-aware Muon**。(若 eSEN 用 FSDP/ZeRO 分片优化器状态,Muon 需要分片感知变体,这点要单独处理 —— 本项目没做。)
- **AMP / GradScaler**:HybridMuon 是标准 Optimizer 子类,兼容 GradScaler、`clip_grad_norm_`、梯度累积、`LambdaLR`、checkpoint `state_dict`。
- **⚠ 精度约束(本项目特有,eSEN 按自身对照)**:本项目梯度力训练(二阶反向 + Wigner 构造)**只能 fp32,绝不开 `--amp`**(amp 会让 Wigner 走 Float/Half index_put 崩);`torch.compile` 在二阶反向下也关掉。eSEN 若无这些路径可忽略,但要明确自己哪些路径是 fp32-only。
- **clip_grad_norm 对 Muon 组基本无效**:Muon 更新是 norm 归一化的,全局 grad clip 会被抵消。**有效的 Muon 保护只能作用在"更新本身"上 → 见 §6。**

## 6. 发散保护 + 曾经的死锁 BUG(本项目已修,eSEN 抄修好的版本)

Muon side 有两道守卫(都在 `_muon_step`,可由 `optim.optimizer_params` 配置):

1. **`skip_nonfinite`(NaN/Inf 梯度)**:梯度非有限就跳过该矩阵,**且不动 momentum buffer / 不动 EMA** —— 防止一次瞬时 NaN 永久毒化 buffer(fp16 死亡螺旋)。这条逻辑**正确,照抄**。
2. **`spike_factor`(EMA 相对尖峰)**:某矩阵的 NS-input `g_rms` 跳过自身 EMA 的 `spike_factor` 倍(默认 8×)就跳过该步。自标定,无需绝对阈值。`spike_warmup_steps`(默认 200)前不触发。

### 曾经的死锁 BUG(已修;说明留作经验)

**旧实现**:spike 命中时 `continue`,而 EMA 更新代码在"提交更新"之后才跑 → **跳步时 EMA 不更新**(momentum buffer 也不更新)。
**后果(实测)**:一次 loss 尖峰把一批矩阵 `g_rms` 顶上去 → skip → EMA **冻结在尖峰前的低值**;之后即使梯度回落或持平,`g_rms` 仍高于"冻结低 EMA × 8" → **每步继续 skip → EMA 永远解不冻 → 永久跳步**。矩阵被一个个吸进陷阱,跳步数单调暴涨(30M run:6→55→107→146,最终约 75% 矩阵每步冻结),**不报 NaN、loss 不发散,但力学不动**(cosine 卡 ~0.1)。这是 ratchet 死锁,**降 lr 只降触发概率、不拆雷**。

> 与 `update_scale` 无关:spike 判定基于 NS-输入 `g_rms`,在缩放(ratio/moonlight)**之前**计算 → 用哪种缩放都一样触发。**moonlight 本身没问题,炸的是守卫。**

### 已实现的修复(见 `muon.py` `_muon_step` Guard 2,eSEN 照抄)

- **(a) spike-skip 时也推进 EMA**(用当前 finite 的 `g_rms`):一次性尖峰只动 EMA ~1%(decay 0.99),仍能拦下次真尖峰;但**持续**的尺度上移会把基线拉上去,阈值几步内恢复。
  - **仅 spike 路径这么改;`skip_nonfinite`(NaN)路径绝不更新 EMA、也绝不 force-commit**(NaN g_rms 会毒化基线 / NaN 梯度不能放过)。
- **(b) 连续跳步上限 `spike_max_consecutive_skips`(默认 25)**:同一矩阵连续跳够这么多步,就**强制提交一次并把 EMA 重置为当前 `g_rms`**,硬保证不会无限冻结。设 `None` 则只靠 (a) 恢复。

**实测验证(本项目)**:持续 10× 平台场景下,旧逻辑整段平台 200 步全冻结;新逻辑只跳 3 步就靠 (a) 自动恢复,之后每步正常训练(cap=25 没用上)。

> eSEN 移植要求:spike 守卫**必须带 (a)+(b)**,否则就是把炸点一起搬过去。保守起见首版也可先 `spike_factor: null` 只留 `skip_nonfinite`,基线跑通再开。**健康判据(硬性)**:`SKIPPED`/`spike` 跳步数必须随训练**减少或持平,绝不能单调增长**。

## 7. 配置接线(yml 示例片段)

```yaml
optim:
  lr_initial: 0.0002            # 非矩阵参数(bias/norm/embed)的 AdamW lr
  optimizer: HybridMuon
  optimizer_params:
    weight_decay: 0.001
    update_scale: moonlight     # 用 moonlight,不用 ratio
    muon_lr: 0.0004             # 矩阵的 Muon lr(与上面的 AdamW lr 是两套)
    momentum: 0.95
    nesterov: True
    ns_steps: 5
    betas: [0.9, 0.95]          # AdamW path
    eps: 1.0e-8
    # 发散保护(见 §6,务必用修好的 spike 守卫):
    skip_nonfinite: True
    spike_factor: 8.0           # 或首版先设 null 关掉,只留 skip_nonfinite
    spike_ema_decay: 0.99
    spike_warmup_steps: 200
    spike_max_consecutive_skips: 25   # 死锁硬底线:连跳 25 步就强制提交+重置 EMA(见 §6)
    log_every: 200
  scheduler: LambdaLR
  scheduler_params:
    lambda_type: cosine
    warmup_factor: 0.001
    warmup_epochs: 0.5          # 给 Muon 长一点的 warmup
    lr_min_factor: 0.01
```

trainer 侧需要:检测到 `optimizer == HybridMuon` 时,用 `build_hybrid_muon_param_groups(model, no_weight_decay, weight_decay, adamw_lr=lr_initial, muon_lr=...)` 造 param groups 再实例化 `HybridMuon`,并打一条 "X params on Muon / Y params on AdamW" 的日志(本项目在 `base_trainer.py` 里就是这么接的,可对照)。

## 8. muon_lr 标定参考(本项目经验,供 eSEN 起步)

- moonlight 下更新 RMS ≈ `0.2*lr`,与形状无关 → 一次标定可迁移。
- 本项目在 C=64 小模型上**验证健康的 moonlight muon_lr = 4e-4**(无 NaN、只在 warmup 跳步、NS-input RMS 下行、轨迹 ≈ ratio-1e-3)。
- 不可超界经验值 ~6e-4;偏热的回退值 2e-4。
- **eSEN 不要直接照搬这个数**:moonlight 的可迁移性是对"同一套实现/同一缩放"而言;eSEN 模型规模/初始化/数据不同,**先用 4e-4 做一次短程 sanity(2~3ep),看 loss 是否在 warmup 后平滑下行、跳步是否只在 warmup 出现且不随训练增长**,再决定。
- **判健康的硬指标**:`SKIPPED` 事件应随训练**减少或持平**,绝不能单调增长;`spike` 跳步在 warmup 后应趋近 0。守卫已带恢复(§6)不会再永久冻结,但若 spike 跳步数仍持续上爬 / 频繁出现 `spike RECOVER` 日志 → **lr 真的偏热**,按 §8 降 muon_lr。

## 9. checkpoint 兼容

- HybridMuon 是标准 Optimizer,`state_dict` 含每个 Muon 矩阵的 `momentum_buffer`、`g_rms_ema`,以及 AdamW 的 `exp_avg/exp_avg_sq/step`。正常 save/restore 即可。
- 跨优化器迁移权重(如从 AdamW ckpt 续训 Muon)走非严格 partial load:只 load 模型权重,优化器 state 重建。

## 10. 对齐 checklist(eSEN 实现完自检)

- [ ] Muon 只接 `ndim>=2` 的真矩阵权重;卷积/特殊张量已 reshape 或划给 AdamW(§4)。
- [ ] `no_weight_decay` 语义复用 eSEN 自己的,用 `endswith` 匹配(§2)。
- [ ] `update_scale=moonlight`,muon_lr 与 AdamW lr 分开配,scheduler 同时缩放两者(§3、§7)。
- [ ] **spike 守卫带上 §6(a) 的 EMA 修复**(或首版 `spike_factor=null` 关掉),`skip_nonfinite` 照抄。
- [ ] DDP 路径验证各 rank 更新一致(§5)。
- [ ] fp32-only 路径不开 AMP(§5,按 eSEN 自身约束)。
- [ ] 短程 sanity(2-3ep):loss 平滑下行、`SKIPPED` 不随训练增长(§8)。
- [ ] checkpoint save/restore 往返一次验证(§9)。
```
