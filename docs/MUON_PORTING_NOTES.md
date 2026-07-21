# 把训练切换到 Muon(HybridMuon)优化器 —— 移植、稳定化与当前配方

> 两个用途:
> 1. **移植参考**(目标读者:负责 **eSEN** 的 Claude Code)—— 在别处复现本项目的 **HybridMuon**。
> 2. **稳定化总账**(本项目自用)—— 记录**尝试过的改动、真实踩过的坑、以及目前跑 70ep-direct 不崩的配方**。
>
> 本项目权威实现:`src/fairchem/core/common/muon.py`(直接读它做参考)。
> 接线参考:`src/fairchem/core/trainers/base_trainer.py`(搜 `HybridMuon` / `build_hybrid_muon_param_groups` / `_load_hybrid_muon`)。

---

## 0. TL;DR(最重要的 6 条)

1. **HybridMuon = Muon(管所有 ≥2D 权重矩阵) + AdamW(管其余:bias / norm / embedding)**。不是纯 Muon。
2. **`update_scale` 选 `moonlight`(本文档称 moonshot 缩放)**(`0.2*sqrt(max(rows,cols))`),不要用 `ratio`(称 Keller 缩放)。moonshot 让同一个 `muon_lr` 跨模型尺寸可迁移;Keller 的有效 lr 依赖矩阵形状,换模型必须重新调。**命名约定 + "Moonshot=公司 / Moonlight=论文" 的澄清见 §3。**
3. **muon_lr 与 AdamW lr 是两套**:`lr_initial`(yml)= 非矩阵参数的 AdamW lr;`optimizer_params.muon_lr` = 矩阵的 Muon lr。scheduler 同时缩放两者。
4. **两类发散、两种防线**——(a)**瞬时尖峰/死锁**由 spike 守卫兜(§6,反应式,逐矩阵跳步);(b)**深模型收敛后的静默尾段 runaway** 由 **γ 权重衰减**根治(§7,主动式,约束每层输出 RMS)。二者互补,缺一不可。
5. **spike 守卫曾有死锁 bug,移植时必须带修复**(§6)。不修的话:大模型/长训练会"跳步数单调暴涨→大半权重被永久冻结→力/精度学不动但不报错"。
6. **梯度力/二阶反向 + Wigner 等 fp32 敏感路径不要开 AMP**(equiformer 特有,§5;eSEN 按自身精度约束对照)。

---

## 1. 核心设计:HybridMuon

Muon(Newton-Schulz 正交化动量)只对**矩阵**有定义,所以非矩阵参数交给 AdamW。一个 `torch.optim.Optimizer` 子类内部同时跑两条 path,按 param-group 上的 `use_muon` 布尔标志分流:

- `use_muon=True` → Muon 更新,用 `lr, momentum, nesterov, ns_steps, weight_decay`。
- `use_muon=False` → 解耦 AdamW 更新,用 `lr, betas, eps, weight_decay`。

## 2. 参数分组规则(关键,容易抄错)

见 `build_hybrid_muon_param_groups`。规则:

| 参数 | 去向 | weight_decay |
|---|---|---|
| 名字命中 `norm_gain_names`(norm 层的 γ/scale) | AdamW | **`norm_weight_decay`**(§7,opt-in;优先级高于下一行) |
| 名字命中 `no_weight_decay` 集合(bias / norm 增益 / embedding) | AdamW | **0** |
| 其余且 `ndim >= 2`(Linear / 任意张量权重) | **Muon** | wd |
| 其余 1D 漏网的 | AdamW | wd |

注意点:
- 用 **`name.endswith(suffix)`** 匹配,而不是精确等于 —— 这样 DDP/compile 包出来的 `module.` / `_orig_mod.` 前缀仍能命中。eSEN 若有自己的 `no_weight_decay` 约定,直接复用其语义。
- 判据是 `p.ndim >= 2`,所以 **3D/4D 权重也会进 Muon**(本项目 `SO3Linear` 是 `(lmax+1, out, in)` 3D)。eSEN 里要确认哪些是"真矩阵权重",哪些是不该正交化的张量(见 §4)。
- **γ 路由(`norm_gain_names`)优先级最高**:norm 增益默认落在 `no_weight_decay` 集合里(不衰减),γ-wd 要把它们**改路**到"AdamW + `norm_weight_decay`"组,所以这条判定必须**先于** `no_weight_decay` 生效(见 `muon.py:434` 注释 "Norm gains (γ) FIRST")。

## 3. `update_scale`:Keller vs moonshot(决定 lr 可迁移性)

> **命名与澄清(全文档通用,先读这段)**
> - **优化器只有一个,叫 Muon**(本项目封装 = HybridMuon)。所谓 "ratio / moonlight" **不是两个优化器**,只是 `muon.py` 里 `update_scale` 这**一个参数的两个取值**,区别仅在正交化之后那**一行缩放公式**;Newton-Schulz 核、`momentum`、`nesterov`、`ns_steps`、hybrid 划分,两者**完全相同**。
> - **本文档命名约定**:`update_scale="ratio"` 一律称 **Keller** 缩放(Keller Jordan 原版 Muon 的默认);`update_scale="moonlight"` 一律称 **moonshot** 缩放。
>   ⚠ **代码 / 配置 / 文件名里的字符串仍是 `"ratio"` / `"moonlight"`,不要改代码**——"Keller" / "moonshot" 只是本文档对这两个模式的叫法,不是代码里的值。
> - **"Moonshot" ≠ "Moonlight",别混**:**Moonshot AI(月之暗面)是公司名**;**Moonlight(arXiv:2502.16982)是那篇论文 / 那个模型的名字**。moonshot 缩放的公式与 §7 的 γ-wd 都出自 **Moonlight 这篇论文**;`update_scale` 取的字符串恰好是 `"moonlight"`,而本文给这个模式起的昵称是 "moonshot"。**正式引用一律写 Moonlight (arXiv:2502.16982)**,别写成 Moonshot。

正交化后 NS 输出 ~半正交,逐元素 RMS ≈ `1/sqrt(max(rows,cols))`。两种缩放把它变成实际步长:

- **Keller**(`update_scale="ratio"`,Keller Jordan):`scale = max(1, rows/cols)**0.5` → 更新 RMS ≈ `lr/sqrt(fan_in)`,**形状相关** → 换 C / ffn_hidden / 深度后 muon_lr 必须重调。
- **moonshot**(`update_scale="moonlight"`,出自 Moonlight,arXiv:2502.16982):`scale = 0.2*sqrt(max(rows,cols))` → 更新 RMS ≈ `0.2*lr` 对**每个矩阵都一样**,与形状无关 → **muon_lr 标定一次即可跨尺寸复用**。

**推荐 moonshot(即 `update_scale="moonlight"`)。** 两种模式量纲不同,切换模式必须重标 muon_lr(Keller 的 lr 拿到 moonshot 没意义,反之亦然)。

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
- **⚠ 精度约束(本项目特有,eSEN 按自身对照)**:本项目梯度力训练(二阶反向 + Wigner 构造)**只能 fp32,绝不开 `--amp`**(amp 会让 Wigner 走 Float/Half index_put 崩);`torch.compile` 在二阶反向下也关掉。direct(直接力)阶段则**开 `--amp`**。eSEN 若无这些路径可忽略,但要明确自己哪些路径是 fp32-only。
- **clip_grad_norm 对 Muon 组基本无效**:Muon 更新是 norm 归一化的,全局 grad clip 会被抵消。**有效的 Muon 保护只能作用在"更新本身"上 → 见 §6、§7。**

## 6. 反应式防线:spike 守卫 + 曾经的死锁 BUG

Muon side 有两道**反应式**守卫(都在 `_muon_step`,可由 `optim.optimizer_params` 配置):

1. **`skip_nonfinite`(NaN/Inf 梯度)**:梯度非有限就跳过该矩阵,**且不动 momentum buffer / 不动 EMA** —— 防止一次瞬时 NaN 永久毒化 buffer(fp16 死亡螺旋)。这条逻辑**正确,照抄**。
2. **`spike_factor`(EMA 相对尖峰)**:某矩阵的 NS-input `g_rms` 跳过自身 EMA 的 `spike_factor` 倍(默认 8×)就跳过该步。自标定,无需绝对阈值。`spike_warmup_steps`(默认 200)前不触发。
3. **`spike_abs_threshold`(绝对 runaway backstop)**:`g_rms` 越过一个**绝对**阈值(如 4.0)也跳,连跳 `spike_abs_max_consecutive`(如 10)则中止。这是 γ-wd(§7)背后的**硬安全网**——相对尖峰守卫在"基线被慢慢拉高"时会自缴械(见下),绝对阈值不会。

### 曾经的死锁 BUG(已修;说明留作经验)

**旧实现**:spike 命中时 `continue`,而 EMA 更新代码在"提交更新"之后才跑 → **跳步时 EMA 不更新**(momentum buffer 也不更新)。
**后果(实测)**:一次 loss 尖峰把一批矩阵 `g_rms` 顶上去 → skip → EMA **冻结在尖峰前的低值**;之后即使梯度回落或持平,`g_rms` 仍高于"冻结低 EMA × 8" → **每步继续 skip → EMA 永远解不冻 → 永久跳步**。矩阵被一个个吸进陷阱,跳步数单调暴涨(30M run:6→55→107→146,最终约 75% 矩阵每步冻结),**不报 NaN、loss 不发散,但力学不动**(cosine 卡 ~0.1)。这是 ratchet 死锁,**降 lr 只降触发概率、不拆雷**。

> 与 `update_scale` 无关:spike 判定基于 NS-输入 `g_rms`,在缩放(Keller/moonshot)**之前**计算 → 用哪种缩放都一样触发。**moonshot 本身没问题,炸的是守卫。**

### 已实现的修复(见 `muon.py` `_muon_step` Guard 2,eSEN 照抄)

- **(a) spike-skip 时也推进 EMA**(用当前 finite 的 `g_rms`):一次性尖峰只动 EMA ~1%(decay 0.99),仍能拦下次真尖峰;但**持续**的尺度上移会把基线拉上去,阈值几步内恢复。
  - **仅 spike 路径这么改;`skip_nonfinite`(NaN)路径绝不更新 EMA、也绝不 force-commit**(NaN g_rms 会毒化基线 / NaN 梯度不能放过)。
- **(b) 连续跳步上限 `spike_max_consecutive_skips`(默认 25)**:同一矩阵连续跳够这么多步,就**强制提交一次并把 EMA 重置为当前 `g_rms`**,硬保证不会无限冻结。设 `None` 则只靠 (a) 恢复。

**实测验证(本项目)**:持续 10× 平台场景下,旧逻辑整段平台 200 步全冻结;新逻辑只跳 3 步就靠 (a) 自动恢复,之后每步正常训练(cap=25 没用上)。

> eSEN 移植要求:spike 守卫**必须带 (a)+(b)**,否则就是把炸点一起搬过去。保守起见首版也可先 `spike_factor: null` 只留 `skip_nonfinite`,基线跑通再开。**健康判据(硬性)**:`SKIPPED`/`spike` 跳步数必须随训练**减少或持平,绝不能单调增长**。

## 7. 主动式防线:γ 权重衰减(norm 增益上的 weight decay)

> **这是"深模型 + Muon 收敛后静默尾段 runaway"的根因修复**,和 §6 的反应式跳步是两回事。§6 只能拦"某矩阵这一步的更新异常";它拦不住"整网输出 RMS 在几个 epoch 里缓慢无界爬升"这种慢漂移(慢漂移不触发 8×EMA 相对阈值,反而会把 EMA 基线一起抬高 → 守卫自缴械)。γ-wd 直接约束那个会漂的量。

### 为什么有效(Moonlight,arXiv:2502.16982 §"weight decay on RMSNorm gamma")

Muon 更新是**梯度幅度盲**的(正交化把幅度信息抹掉,无 AdamW 那种逐坐标 `/√v̂` 自适应阻尼)。深模型训到后期,某些层的 norm 增益 γ 会持续增大 → 该层**输出 RMS 无界上升** → 下游梯度/激活被放大 → Muon 又以固定步长照推 → 正反馈 → runaway。**对 γ 加一个小 weight decay,等于给每层输出 RMS 装了个回拉弹簧**,把这条正反馈掐断。Moonlight 明确称之为 "crucial for stability, prevents excessively high output RMS"。

### 代码实现(opt-in,默认关 = 字节级不变)

- `build_hybrid_muon_param_groups(..., norm_gain_names, norm_weight_decay)`(`muon.py:401`):把 `norm_gain_names` 里的 γ 从"AdamW no-decay"**改路**到新组 `adamw_norm_decay`(`use_muon=False, weight_decay=norm_weight_decay`)。**该判定优先于 `no_weight_decay`**(γ 默认在 no-decay 集里,必须先被这条override 抢走)。`norm_gain_names=None` → 行为与旧版**字节一致**(纯 opt-in)。
- `base_trainer._load_hybrid_muon`(`base_trainer.py:797-844`):当 `optim.optimizer_params.norm_weight_decay` **被设**时,遍历 `named_modules()`,凡**类名含 "norm"** 的模块,取其 `.weight`(即 γ/scale,**不含 bias**)收进 `norm_gain_names`;否则集合为空 → 不改变任何行为。master 上额外打一条 `"N norm-gain (γ) tensors moved to AdamW with weight_decay=..."` 日志。
- 配置开关:`optim.optimizer_params.norm_weight_decay: 1e-3`。

### 与 §6 的分工(记住这张表)

| | §6 spike 守卫 | §7 γ 权重衰减 |
|---|---|---|
| 类型 | 反应式(事后跳步) | 主动式(约束根源量) |
| 拦的失效 | 瞬时尖峰 / NaN / 死锁 | **收敛后输出 RMS 慢漂移 → 尾段 runaway** |
| 对慢漂移 | **无效**(甚至自缴械) | 有效 |
| 角色 | 安全网 / backstop | 根因修复 |

## 8. 踩坑总账:真实遇到过的发散(N2L2C64 / N7L4C128)

按"能不能自愈/会不会报错"分类,方便对号入座:

1. **spike 守卫死锁**(§6)—— 静默冻结,不报错、不发散、力学不动。**已由 (a)+(b) 修复。**
2. **Keller 缩放 N2L2C64 NaN 级联** —— 硬 NaN 崩溃(Keller 有效 lr 形状相关、更易过热)。**改用 moonshot + 合理 muon_lr 解决。**
3. **moonshot N7L4C128 静默尾段 runaway(核心难点)** —— 深 30M 模型上:`muon_lr=4e-4` 约 **ep9.6** runaway;`2e-4` **健康到 ep20、ep21 才 runaway**;经验规律**"lr 减半 ≈ 存活翻倍"**。特征:EMA spike 守卫自缴械、**无 NaN → 不崩溃 → run 会"跑完"但已被 lobotomize**(力精度废掉)。**这正是 §7 γ-wd 针对的失效。**
4. **schemeS:Muon 低 lr direct 二阶段续训(从 ep20 峰值 CKPT 起)** —— 两个独立失败:
   - **(a) 真发散**:train loss 0.89(ep1)→0.94(ep2)→5.32(ep3)→AMP GradScaler 在 ep5.514 塌到 0,`RuntimeError`。**"冷启动(mlr 5e-5 < 7.8e-5)+ 新动量 → 安全"的假设被证伪**(ep5.5 中途照炸)。
   - **(b) 仪表失效**:`best_checkpoint` 从未写出(只有 `checkpoint.pt`),因为 val `forces_mae` 从 ep1 就是 NaN —— 很可能是 **Muon+DeNS 的早期真实不稳定**,不是无害的 amp 测量伪影。
   - 结论:**Muon 低 lr 直接续训 = 死路。**
5. **对照:AdamW 低 lr refine 从 ep20 起 —— 成功**(全程 15ep 不崩,best@ep10.81 forces_mae 0.026369,cos 0.7602)。**AdamW 的逐坐标 `/√v̂` 自阻尼能守住 Muon 守不住的盆地。** 这条对照直接催生了 §7:**给 Muon 也装一个"自阻尼",就是 γ-wd。**

> 一句话:**深模型的 Muon 难点不是"会不会 NaN 崩",而是"会不会静默 lobotomize"**——后者不报错、best_ckpt 甚至照存,最坑。健康判据永远看 `max NS-input RMS` 是否上爬、val cosine 是否掉头,而不是"有没有崩"。

## 9. 当前配方:STABILIZED 70ep-direct(四层防线)+ 现状

**目标**:把被"25ep 早衰帽"钉死的欠训解开,对齐 official 的 70ep direct 训练时长,同时扛住 ep15–25 危险带。
**配置**:`experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_..._epochs@70-bs@32x16_hybridmuon-moonlight-mlr@2e-4-alr@2e-4-wd@1e-3-normwd@1e-3-warmup@0.5-minf@0.01_dens-no-stress_loss-e5-f10-s100_STABILIZED.yml`
**muon_lr = 2e-4**(moonshot,即 `update_scale="moonlight"`);非矩阵 AdamW `lr_initial` 也 = 2e-4。

### 四层防线(从根因到硬兜底)

| 层 | 机制 | 配置 | 作用 |
|---|---|---|---|
| ① 根因修复 | **γ 权重衰减**(§7) | `norm_weight_decay: 1e-3` | 约束每层输出 RMS,掐断尾段 runaway 的正反馈 |
| ② γ-wd 所许可的松绑 | muon_lr 回抬 + 松尾 | `muon_lr 1.5e-4→2e-4`,`lr_min_factor 0.001→0.01` | γ-wd 兜住 RMS 后,不必再靠"压 lr + 陡尾早衰"逃离危险带;让后 45 ep 真正以有效 lr 训练 |
| ③ 相对尖峰守卫 | 逐矩阵跳步(§6) | `spike_factor 8.0` + `ema_decay 0.99` + `warmup 200` + `max_consecutive 25` | 拦瞬时尖峰,自愈 |
| ④ 绝对硬兜底 | runaway backstop | `spike_abs_threshold 4.0` + `abs_max_consecutive 10` + `skip_nonfinite` | γ-wd 若压不住则中止,`best_ckpt` = runaway 前峰值 |

配套:更长更缓 warmup(`warmup_epochs 0.5`,`warmup_factor 1e-3`)喂 Muon 进满 lr。

### 升级阶梯(若 ep15–25 仍尖峰)

**先** `norm_weight_decay 1e-3→2e-3`(加 γ 阻尼,最便宜);**再** `muon_lr 2e-4→1.5e-4`。**不要**重新收陡尾(那只是复刻 25ep 天花板)。观察窗:`max NS-input RMS` 是否上爬 / val cosine 是否掉头。

### ⚠ 现状(诚实标注,勿高估)

- run:`2026-07-07-01-33-52-muon_N7L4C128_direct_70ep_moonlight_mlr2e-4_normwd1e-3_STABILIZED`。
- **只经验验证到 ep4**:日志干净停在 epoch 4.00(val forces_mae **0.0345** / cos **0.6884** / loss 1.16),结尾是一次正常 val 评估,**无 traceback、无发散**;spike 守卫在正常工作(streak-1 跳步、立即恢复)。
- 之后作业**在 ep4 停了/被杀,不是崩、是没继续跑**(checkpoint 自 07-07 后未再更新)。
- **关键的 ep15–25 危险带根本还没进去**——那正是 35ep-direct 在 ep21 runaway、schemeS 在 ep5.5 炸的地方。**所以"γ-wd 让 70ep 不崩"目前是"设计到位 + ep4 前实测健康",尚未经验证明。** 需从 ep4 的 checkpoint 续跑去真正撞危险带才能定论。

## 10. 配置接线(yml 示例片段)

```yaml
optim:
  lr_initial: 0.0002            # 非矩阵参数(bias/norm/embed)的 AdamW lr
  optimizer: HybridMuon
  optimizer_params:
    weight_decay: 0.001
    norm_weight_decay: 0.001    # §7 γ-wd(opt-in;不设则字节级不变)。深模型/长训练强烈建议开
    update_scale: moonlight     # moonshot 缩放(不要用 Keller/ratio);代码里字符串仍写 "moonlight"
    muon_lr: 0.0002             # 矩阵的 Muon lr(与上面的 AdamW lr 是两套)
    momentum: 0.95
    nesterov: True
    ns_steps: 5
    betas: [0.9, 0.95]          # AdamW path
    eps: 1.0e-8
    # 反应式发散保护(§6,务必用修好的 spike 守卫):
    skip_nonfinite: True
    spike_factor: 8.0           # 或首版先设 null 关掉,只留 skip_nonfinite
    spike_ema_decay: 0.99
    spike_warmup_steps: 200
    spike_max_consecutive_skips: 25   # 死锁硬底线:连跳 25 步就强制提交+重置 EMA(§6)
    spike_abs_threshold: 4.0          # 绝对 runaway backstop(§6.3)
    spike_abs_max_consecutive: 10
    log_every: 200
  scheduler: LambdaLR
  scheduler_params:
    lambda_type: cosine
    warmup_factor: 0.001
    warmup_epochs: 0.5          # 给 Muon 长一点的 warmup
    lr_min_factor: 0.01         # 松尾;γ-wd 上了以后不要再收陡(§9)
```

trainer 侧需要:检测到 `optimizer == HybridMuon` 时,走 `_load_hybrid_muon` → 用 `build_hybrid_muon_param_groups(model, no_weight_decay, weight_decay, adamw_lr=lr_initial, muon_lr=..., norm_gain_names=..., norm_weight_decay=...)` 造 param groups 再实例化 `HybridMuon`,并打 "X params on Muon / Y params on AdamW /(若开)N norm-gain γ moved" 的日志(本项目在 `base_trainer.py` 里就是这么接的,可对照)。

## 11. muon_lr 标定参考(本项目经验,供 eSEN 起步)

- moonshot 下更新 RMS ≈ `0.2*lr`,与形状无关 → 一次标定可迁移。
- 本项目在 C=64 小模型上**验证健康的 moonshot muon_lr = 4e-4**(无 NaN、只在 warmup 跳步、NS-input RMS 下行、轨迹 ≈ Keller-1e-3)。
- **深 30M(N7L4C128)则要更冷**:`4e-4` 约 ep9.6 runaway、`2e-4` 健康到 ep20(§8-3)。**深度不像宽度那样可迁移**——moonshot 保证的是"跨宽度",不是"跨深度"。深模型先按 `2e-4` 起,并**务必开 γ-wd**(§7)。
- 不可超界经验值(小模型)~6e-4;深模型偏热回退值 1.5e-4。
- **eSEN 不要直接照搬这些数**:先用一个偏冷值做短程 sanity(2~3ep),看 loss 是否在 warmup 后平滑下行、跳步是否只在 warmup 出现且不随训练增长,再决定。
- **判健康的硬指标**:`SKIPPED` 事件应随训练**减少或持平**,绝不能单调增长;`spike` 跳步在 warmup 后应趋近 0;`max NS-input RMS` 不应在中后期持续上爬。守卫已带恢复(§6)不会再永久冻结,但 RMS 慢爬 / val cosine 掉头 = 尾段 runaway 前兆 → 按 §9 阶梯加 γ-wd / 降 muon_lr。

## 12. checkpoint 兼容

- HybridMuon 是标准 Optimizer,`state_dict` 含每个 Muon 矩阵的 `momentum_buffer`、`g_rms_ema`,以及 AdamW 的 `exp_avg/exp_avg_sq/step`。正常 save/restore 即可。
- 跨优化器迁移权重(如从 AdamW ckpt 续训 Muon)走非严格 partial load:只 load 模型权重,优化器 state 重建。
- **续训注意**:从 ep20 峰值 CKPT 用 **Muon 低 lr** 直接续 direct 是**死路**(§8-4);要续训基本盘就换 **AdamW 低 lr refine**(§8-5),要在 Muon 里继续训就走 §9 的 γ-wd 全程配方,而不是"续一段低 lr"。

## 13. 对齐 checklist(eSEN 实现完自检)

- [ ] Muon 只接 `ndim>=2` 的真矩阵权重;卷积/特殊张量已 reshape 或划给 AdamW(§4)。
- [ ] `no_weight_decay` 语义复用 eSEN 自己的,用 `endswith` 匹配(§2)。
- [ ] `update_scale=moonlight`(moonshot 缩放),muon_lr 与 AdamW lr 分开配,scheduler 同时缩放两者(§3、§10)。
- [ ] **spike 守卫带上 §6(a)+(b) 的修复**(或首版 `spike_factor=null` 关掉),`skip_nonfinite` 照抄;绝对 backstop `spike_abs_threshold` 建议一并带(§6.3)。
- [ ] **深模型 / 长训练开 γ-wd `norm_weight_decay`**(§7);确认 γ 收集覆盖到 eSEN 的 norm 层(类名含 "norm" 的 `.weight`)。
- [ ] DDP 路径验证各 rank 更新一致(§5)。
- [ ] fp32-only 路径不开 AMP(§5,按 eSEN 自身约束)。
- [ ] 短程 sanity(2-3ep):loss 平滑下行、`SKIPPED` 不随训练增长、`max NS-input RMS` 不中后期上爬(§8、§11)。
- [ ] checkpoint save/restore 往返一次验证(§12)。
