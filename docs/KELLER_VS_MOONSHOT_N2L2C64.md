# keller vs moonshot —— N2L2C64 15+5ep Muon 变体消融

> 日期:2026-07-21 · 口径:5% as-predicted(unique_prototypes F1)+ 全量 103 声子 κ_SRME
>
> **命名约定(全项目别名)**:Muon 是**一个**优化器,`update_scale` 是**一个**参数、有两个取值——
> **keller ≡ `update_scale="ratio"`**(Keller Jordan 默认缩放 `max(1, rows/cols)**0.5`);
> **moonshot ≡ `update_scale="moonlight"`**(Moonshot AI 的 Moonlight 论文 arXiv:2502.16982 缩放 `0.2*sqrt(max(rows,cols))`)。
> "keller / moonshot" 仅为**文档昵称**;代码、config、文件名里的字符串仍写 `ratio` / `moonlight`。
> NS 核、动量、nesterov、ns_steps、hybrid 拆分全部共享,**唯一区别是这一行 rescale**。

---

## TL;DR

- **本轮 N2L2C64 A/B:keller 综合胜出 CPS +0.0168**(0.7470 vs 0.7302)。
- **但赢在一阶量,不在热导率**:keller 的 F1 高 +0.0319、RMSD 更小,而 **κ_SRME 反而微输**(moonshot κ 略好)。
- 这与外部非等变团队 ~50M 的结论**殊途同归、机制相反**——他们的 keller 是靠 **κ** 赢、F1 持平;我们的 keller 靠 **F1/RMSD** 赢、κ 微输。→ **"moonshot 损伤高阶 PES 曲率(κ)"的特征在 N2L2C64 这个小模型上没有复现,疑似 size-dependent。**
- **本 A/B 完全没有测到稳定性轴**:N2L2C64 太小,两臂都不会尾段 runaway。因此外部团队"稳定性来自 weight decay(与缩放正交)、不是来自 moonshot 缩放"的重构,**在本文档里未被检验**,属于 N7L4C128 深模型的问题(见 §5)。

---

## 1. 目的

在**完全相同**的 N2L2C64 / direct 15ep → grad 5ep / loss e:f:s=5:10:100 配方下,**唯一变量为 `update_scale`**,比较两者综合分 CPS:

| | keller(`ratio`) | moonshot(`moonlight`) |
|---|---|---|
| ckpt 目录 | `2026-07-20-04-01-04-kappaAB_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_maxatoms150_bs16x8x4_loss-e5f10s100` | `2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100` |
| `update_scale` | `ratio` | `moonlight` |
| max learning rate | 6.6e-4 | 1.5e-4 |
| 家族 | 原生 eqV3-fork(OCPCalculator) | 原生 eqV3-fork(OCPCalculator) |

> lr 不同是**故意匹配**:moonshot 更新 RMS ≈ `0.2·lr` 与形状解耦,keller 更新 RMS = `lr/√fan_in`;二者的等效换算约 2.5×(keller 1e-3 ≈ moonshot 4e-4),所以 keller 6.6e-4 与 moonshot 1.5e-4 属同一档有效步长,不是混淆变量。
> 两者均为原生 eqV3-fork、测的都是 gradft 段 `best_checkpoint.pt`,声子固定 103 结构、n_overlap=103 有效。

## 2. 分项结果

CPS = 0.5·F1ₙ + 0.4·κ_SRMEₙ + 0.1·RMSDₙ,其中
F1ₙ=F1;κ_SRMEₙ=max(0, 1−SRME/2);RMSDₙ=clamp((0.15−RMSD)/0.15, 0, 1)。
F1/RMSD 取 5% as-predicted 的 unique_prototypes 口径;κ_SRME 取全量 103 声子真值。

| 分项 | 权重 | **keller(ratio)** | **moonshot(moonlight)** | Δ(keller−moonshot) |
|---|---|---|---|---|
| F1 | 0.5 | **0.7844** 🥇 | 0.7525 | **+0.0319** |
| κ_SRME ↓ | — | 0.4718 | **0.4638** 🥇 | +0.0080 |
| κ_SRMEₙ | 0.4 | 0.7641 | **0.7681** 🥇 | −0.0040 |
| RMSD ↓ | — | **0.0763** 🥇 | 0.0800 | −0.0037 |
| RMSDₙ | 0.1 | **0.4912** 🥇 | 0.4667 | +0.0245 |
| **CPS** | | **0.7470** 🥇 | 0.7302 | **+0.0168** |

**加权贡献拆解:**

| | 0.5·F1 | 0.4·κ_SRMEₙ | 0.1·RMSDₙ | **= CPS** |
|---|---|---|---|---|
| keller(ratio) | 0.3922 | 0.3056 | 0.0491 | **0.7470** |
| moonshot(moonlight) | 0.3763 | 0.3072 | 0.0467 | **0.7302** |

> 备注:若改用 full_test_set F1 口径,两者 F1 = 0.7708 / 0.7445,CPS = 0.7402 / 0.7262,keller 仍胜 +0.0140。本文档主表采用 unique_prototypes 口径以与 CPS 汇总表保持一致。

## 3. 本轮结论

1. **keller(ratio)综合胜出 +0.0168(CPS 0.7470 vs 0.7302)**,明确超过 κ 噪声地板(≈0.002)与 F1 噪声地板(≈0.003)。
2. **胜负手是一阶量,不是热导率**:
   - moonshot(moonlight)在 κ_SRME 上微弱领先 0.0080(κₙ +0.0040 → 对 CPS 仅 +0.0016);
   - 但 keller(ratio)在 **F1 高 0.0319**(权重 0.5 → +0.0160)+ **RMSD 更小**(RMSDₙ +0.0245,权重 0.1 → +0.0025),两项一阶量合计 +0.0185,把 κ 的小劣势彻底盖过。
3. **物理解读**:Muon-ratio 让能量面/受力(E/F 一阶,决定 F1、RMSD)更准,代价是二三阶曲率(FC2/FC3,决定 κ)几乎持平、略退一点点——净效应对 CPS 为**正向**。
4. **选型陷阱**:若只看声子会误判成"moonshot 更好";**纳入一阶量后 keller 才是这组 15+5ep 的更优缩放**。

## 4. 与外部非等变 MLIP 团队的交叉比对(2026-07-21 W30 周报)

另一支**非等变** MLIP 团队在 ~50M 同尺寸("M")两臂上跑了同样的 keller-vs-moonshot 比较,结论方向一致、机制相反,值得并列。

| | 本项目 N2L2C64(等变 eqV3) | 外部团队 ~50M(非等变) |
|---|---|---|
| 谁赢 CPS | **keller** +0.0168 | **keller** ~+0.05 |
| F1 | keller **高** +0.0319 | 基本持平(0.858 vs 0.863) |
| κ_SRME | keller **微输**(0.4718 vs 0.4638) | keller **大胜**(0.43 vs 0.68–0.73) |
| 赢的来源 | **一阶量(F1/RMSD)** | **κ(高阶曲率)** |
| moonshot 是否带 µP | 否(纯 update_scale 变量) | 是(moonshot+µP,κ 归因不纯) |
| 稳定性是否触发 | 否(模型太小) | 是(keller 在 ~140M L 失稳) |

**读法:**

1. **"keller ≥ moonshot on CPS" 被两个架构、两个尺度独立确认**——这是稳的结论,抬高我们后续默认选 keller 的先验。
2. **但"moonshot 损伤 κ"的特征在 N2L2C64 没复现,甚至反号**。最可能的解释是 **size-dependent**:moonshot 的形状解耦步长(`0.2·lr` 常数)要到宽度/深度足够大、矩阵内谱各向异性真正开始主导时,才会开始拖累高阶 PES 曲率。N2L2C64 太小,谱各向异性不足以让这个机制显形。→ **我们自己的大模型 A/B 才是决定性检验,不能拿 N2L2C64 的 κ 反号去否定外部团队。**
3. 外部团队的 moonshot 臂**捆绑了 µP**,κ 归因不是单变量干净;我们这轮虽是纯 update_scale 变量,但 stage-1 有 lr 起点混淆(ratio 段 @2e-3 未与 1e-3 严格匹配)。**两边都不是无懈可击的单变量,但都指向同一方向。**

## 5. weight-decay 重构:本 A/B 未触及的稳定性轴

外部团队最重要的一条,是把**稳定性的来源从"缩放模式"剥离到"weight decay":**

- keller 在 M 稳、到 L(~140M)失稳:grad-norm 中后期爬升、spike-skip/回滚、权重幅值在**深层**无界增长——正是 Moonlight 论文的根因(keller 缩放 → 规模上权重无界增长)。
- 他们的**修法 = weight decay 0.1**(原默认弱了 ~400×),**施加在 keller 上**即根治稳定性(grad-norm 封顶、跑过旧崩溃点),代价是上游 **force MAE +15–20%**。
- **关键:稳定性来自 weight decay,它与 `update_scale` 正交,不来自 moonshot 缩放的切换。** 所以 **"keller + 强 wd"可能同时拿到 keller 的 κ/一阶优势 + 稳定性**,压过 moonshot——这才是可能的最优象限。

**对本文档的边界说明(重要,避免过度外推):**

- **N2L2C64 这轮 A/B 完全没有测到稳定性轴**——两臂都太小、都不会尾段 runaway。因此本文档的 CPS 胜负**只反映"配方对不对",不反映"能不能在深模型上稳住"**。
- 稳定性 + wd 重构是 **N7L4C128 深模型**的问题,属另一条实验线,**不能用本文档的结论去回答**。
- 我们当前 config 的矩阵 `weight_decay: 0.001`,比 Moonlight 的 0.1 **弱 100×**;而且我们一直只用 γ-wd(`norm_weight_decay=1e-3`)去压 N7L4C128 的 runaway——外部数据说真正的杠杆是**矩阵 wd**,我们很可能**既欠剂量又打错了靶**。

## 6. 下一步消融实验建议(按优先级)

1. **【最高优先·决定性】把这套 A/B 搬到大模型跑一遍(N7L4C128 或能让 moonshot κ 特征显形的尺度)。**
   目的:验证 §4 的 size-dependent 假设——大模型上 keller 的赢法会不会从"靠 F1"翻转成外部团队的"靠 κ"。这是唯一能把 N2L2C64 的 κ 反号与外部 κ 大胜调和的实验。

2. **【关键开放问题】矩阵 wd 扫描 0.02 / 0.05 / 0.1,施加在 keller 上,跑 N7L4C128。**
   同时测三样:(a)稳定性(grad-norm 尾段、有没有跑过旧崩溃点);(b)κ_SRME;(c)force MAE 代价。
   **枢纽问题**:强 wd 在稳住的同时会不会侵蚀 keller 的 κ/一阶优势?外部团队只报了 force +15–20% 的代价,**没报 wd 对 κ 的影响**。要看 **CPS 净值**,并扫出**最小可稳定 wd**——按 [[mp-sota-gap-significance]],15–20% 的 force 退化是灾难级,除非 κ 的 CPS 增益能反超。

3. **【选型默认】把文档层面的"moonshot = 默认"从既定结论降级为"keller + 强 wd 是活跃且可能更优的候选"。**
   依据:N2L2C64(本轮)+ 外部 ~50M 两个独立证据都是 keller ≥ moonshot on CPS;稳定性可由矩阵 wd 单独买。待 §6.1 大模型 A/B + §6.2 wd 扫描落地后再定稿到 `docs/MUON_PORTING_NOTES.md`。

4. **【便宜的稳健性检查】N2L2C64 keller 的 κ 微劣是否只是噪声?**
   κ 差 0.0080 vs 噪声地板 ~0.002:量级上是真的,但很小。若换 seed / 换声子子集代价低,补一个重复点,确认 κ 反号是稳的信号而不是单点抖动——这会直接影响 §4.2 的 size-dependent 解读强度。

## 7. 数据溯源

- keller(ratio)弛豫:`.../matbench/relaxation/muon_ratio_n2l2c64_gradft_5ep_e5f10s100/{metrics.json,rmsd.txt}`(2026-07-21 12:08 产出)
- keller(ratio)声子:`.../matbench/phonon/muon_ratio_n2l2c64_gradft_5ep_e5f10s100/`(SRME=0.4718,n_overlap=103)
- moonshot(moonlight)弛豫:`.../matbench/relaxation/abl_n2l2c64_gradft_loss-e5f10s100/{metrics.json,rmsd.txt}`
- moonshot(moonlight)声子:`.../matbench/phonon/abl_n2l2c64_gradft_loss-e5f10s100/`(SRME=0.4638,n_overlap=103)
- 提交脚本:`Matbench-tools/{relaxation,phonon}/submit_muon_ratio_n2l2c64_gradft_5ep*.sh`、`submit_abl_e5f10s100.sh`
- 外部团队周报交叉比对来源:记忆 `external-mlip-muon-wd-finding`(W30, 2026-07-21)。
