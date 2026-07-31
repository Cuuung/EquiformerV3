# keller vs moonshot —— N2L2C64 Muon 变体消融

> 初版 2026-07-21 · **干净重测 2026-07-27(本版,取代旧结论)** · 口径:5% as-predicted(unique_prototypes F1)+ 全量 103 声子 κ_SRME
>
> **命名约定(全项目别名)**:Muon 是**一个**优化器,`update_scale` 是**一个**参数、有两个取值——
> **keller ≡ `update_scale="ratio"`**(Keller Jordan 默认缩放 `max(1, rows/cols)**0.5`);
> **moonshot ≡ `update_scale="moonlight"`**(Moonshot AI 的 Moonlight 论文 arXiv:2502.16982 缩放 `0.2*sqrt(max(rows,cols))`)。
> "keller / moonshot" 仅为**文档昵称**;代码、config、文件名里的字符串仍写 `ratio` / `moonlight`。
> NS 核、动量、nesterov、ns_steps、hybrid 拆分全部共享,**唯一区别是这一行 rescale**。

---

## ⚠️ 本版重要更正(2026-07-27)

初版(2026-07-21)的 keller 臂用的是 `2026-07-20` 的 ckpt,它的 **direct 基座与 moonshot 臂(消融A)不是同一个** ——
基座差异被冒充成了"变体增益",导致两条错误印象:**(a)** keller 赢 +0.0168、**(b)** 胜负手是 F1、keller 反而 κ 微输。

2026-07-27 重训了一条**基座对齐**的 keller(`keller_n2l2c64_gradft5ep_from_abl_moonlight15ep`):
用**消融A 那个 moonlight 15ep direct 基座**,gradft 段唯一变量 `moonshot→keller`(lr 1.5e-4→6.6e-4)。真相是:

- **keller 仍胜,但净增益只有 +0.0047**(不是 +0.0168);
- **胜负手是 κ_SRME**(keller 0.4459 < moonshot 0.4638),**不是 F1**;
- **F1 打平**(0.7529 vs 0.7525,噪声内)。初版看到的 F1 +0.0319 全部来自基座差异。

**这条更正把本文档与外部团队的比对从"机制相反"改成"机制一致"**(两边都是 keller 靠 κ 赢、F1 持平)——见 §4。

---

## TL;DR

- **基座对齐后:keller 综合胜出 CPS +0.0047**(0.7349 vs 0.7302),超过 κ 噪声地板(≈0.002)。
- **赢在热导率,不在一阶量**:keller 的 **κ_SRME 更低 −0.0179**(κₙ +0.0089→CPS +0.0036,胜负手);**F1 打平**(+0.0004);RMSD 微弱更小。
- 这与外部非等变团队 ~50M 的结论**同方向、且机制也一致**——他们的 keller 也是靠 **κ** 赢、F1 持平。→ **"moonshot 损伤高阶 PES 曲率(κ)"的特征在 N2L2C64 上确实复现了**(初版误判为"没复现/反号",病根是基座污染,不是 size-dependent)。
- **本 A/B 仍完全没有测到稳定性轴**:N2L2C64 太小,两臂都不会尾段 runaway。外部团队"稳定性来自 weight decay(与缩放正交)"的重构在本文档里**未被检验**,属 N7L4C128 深模型的问题(见 §5)。

---

## 1. 目的

在**完全相同**的 N2L2C64 / direct 15ep → grad 5ep / loss e:f:s=5:10:100 配方、**且同一 direct 基座**下,**唯一变量为 `update_scale`**,比较两者综合分 CPS:

| | keller(`ratio`) | moonshot(`moonlight`) |
|---|---|---|
| ckpt 目录 | `2026-07-27-07-21-36-keller_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_from-abl-moonlight15ep-direct_KAPPA-AB` | `2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100`(消融A) |
| direct 基座 | **消融A moonlight 15ep(与右列同一基座)** | 消融A moonlight 15ep |
| `update_scale` | `ratio` | `moonlight` |
| max learning rate | 6.6e-4 | 1.5e-4 |
| 家族 | 原生 eqV3-fork(OCPCalculator) | 原生 eqV3-fork(OCPCalculator) |

> lr 不同是**故意匹配**:moonshot 更新 RMS ≈ `0.2·lr` 与形状解耦,keller 更新 RMS = `lr/√fan_in`;二者等效换算约 2.5×,所以 keller 6.6e-4 与 moonshot 1.5e-4 属同一档有效步长,不是混淆变量。
> **基座对齐是本版相对初版的关键修正**:唯一变量真正落在 `update_scale` 上,不再混入 direct 基座差异。
> 两者均为原生 eqV3-fork、测的都是 gradft 段 `best_checkpoint.pt`(epoch 5.0,`direct_prediction=False` 铁律通过),声子固定 103 结构、n_overlap=103 有效。

## 2. 分项结果(基座对齐)

CPS = 0.5·F1ₙ + 0.4·κ_SRMEₙ + 0.1·RMSDₙ,其中
F1ₙ=F1;κ_SRMEₙ=max(0, 1−SRME/2);RMSDₙ=clamp((0.15−RMSD)/0.15, 0, 1)。
F1/RMSD 取 5% as-predicted 的 unique_prototypes 口径;κ_SRME 取全量 103 声子真值。

| 分项 | 权重 | **keller(ratio)** | **moonshot(moonlight)** | Δ(keller−moonshot) |
|---|---|---|---|---|
| F1 | 0.5 | 0.7529 | 0.7525 | **+0.0004**(打平,噪声内) |
| κ_SRME ↓ | — | **0.4459** 🥇 | 0.4638 | **−0.0179** |
| κ_SRMEₙ | 0.4 | **0.7770** 🥇 | 0.7681 | **+0.0089** |
| RMSD ↓ | — | **0.0786** 🥇 | 0.0800 | −0.0014 |
| RMSDₙ | 0.1 | **0.4761** 🥇 | 0.4667 | +0.0094 |
| **CPS** | | **0.7349** 🥇 | 0.7302 | **+0.0047** |

**加权贡献拆解:**

| | 0.5·F1 | 0.4·κ_SRMEₙ | 0.1·RMSDₙ | **= CPS** |
|---|---|---|---|---|
| keller(ratio) | 0.3765 | 0.3108 | 0.0476 | **0.7349** |
| moonshot(moonlight) | 0.3763 | 0.3072 | 0.0467 | **0.7302** |

**Δ 来源**:κ_SRME **+0.0036**(胜负手)+ RMSD +0.0009 + F1 +0.0002 = **+0.0047**。

> **初版(基座污染)对比留档,作方法学警示**:旧 keller(`muon_ratio_..._2026-07-20` 基座)F1=0.7844、κ=0.4718、RMSD=0.0763、CPS=0.7470;
> 看着 keller 赢 +0.0168、靠 F1(+0.0319)、κ 微输——**这三点全是基座差异冒充的,基座对齐后消失**。教训:变体消融**必须锁死 direct 基座**,否则一阶量口径会被基座质量主导。

## 3. 本轮结论(基座对齐)

1. **keller(ratio)综合胜出 +0.0047(CPS 0.7349 vs 0.7302)**,超过 κ 噪声地板(≈0.002)。增益小但方向稳。
2. **胜负手是热导率(κ),不是一阶量**:
   - keller 在 κ_SRME 上领先 0.0179(κₙ +0.0089 → 对 CPS +0.0036);
   - F1 打平(+0.0004,权重 0.5 → +0.0002),RMSD 微弱(+0.0094,权重 0.1 → +0.0009)。
3. **物理解读**:在同一收敛 direct 基座上,keller 的形状感知步长(`lr/√fan_in`)让 gradft 段更好地压平二三阶曲率(FC2/FC3,决定 κ),而一阶量(E/F,决定 F1/RMSD)两臂基本持平。**这与"moonshot 的形状解耦常数步长略伤高阶曲率"一致。**
4. **选型**:纳入 κ 后 **keller 是这组 15+5ep 的更优缩放**,且赢的来源(κ)与外部团队一致——抬高后续默认选 keller 的先验。

## 4. 与外部非等变 MLIP 团队的交叉比对(2026-07-21 W30 周报,本版更新)

另一支**非等变** MLIP 团队在 ~50M 同尺寸("M")两臂上跑了同样的 keller-vs-moonshot 比较。**基座对齐重测后,两边结论方向一致、机制也一致**:

| | 本项目 N2L2C64(等变 eqV3,**基座对齐**) | 外部团队 ~50M(非等变) |
|---|---|---|
| 谁赢 CPS | **keller** +0.0047 | **keller** ~+0.05 |
| F1 | **基本持平**(0.7529 vs 0.7525) | 基本持平(0.858 vs 0.863) |
| κ_SRME | keller **胜**(0.4459 vs 0.4638) | keller **大胜**(0.43 vs 0.68–0.73) |
| 赢的来源 | **κ(高阶曲率)** | **κ(高阶曲率)** |
| moonshot 是否带 µP | 否(纯 update_scale 变量) | 是(moonshot+µP,κ 归因不纯) |
| 稳定性是否触发 | 否(模型太小) | 是(keller 在 ~140M L 失稳) |

**读法:**

1. **"keller ≥ moonshot on CPS,且赢在 κ" 被两个架构、两个尺度独立确认**——这是稳的结论,抬高我们后续默认选 keller 的先验。
2. **初版的"机制相反 / moonshot 损伤 κ 在小模型没复现"是错的**,病根是基座污染而非 size-dependent。基座对齐后,N2L2C64 上 keller 同样靠 κ 赢,与外部团队机制一致。**"moonshot 形状解耦步长略伤高阶 PES 曲率"这个特征在小模型也成立**,只是量级小(κ 差 0.0179,vs 外部 ~0.25 的量级差),规模越大越显著。
3. 外部团队的 moonshot 臂**捆绑了 µP**,κ 归因不是单变量干净;我们这轮是**纯 update_scale 变量、且基座对齐**,归因比外部更干净——**两边都指向同一方向,我们这条是更硬的单变量证据。**

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

1. **【已部分回答·仍需大模型确认】size-dependent 假设**:初版靠 N2L2C64 的 κ"反号"去质疑外部团队,**已被基座对齐重测证伪**——小模型上 keller 同样靠 κ 赢。剩下的问题不再是"机制会不会翻转",而是"**κ 增益的量级随尺度怎么长**":N2L2C64 只有 0.0179,外部 ~50M 是 ~0.25。**在 N7L4C128 上跑基座对齐的 keller-vs-moonshot,量一量 κ 增益是否随尺度放大**——这才是当前该测的。
   > 注:现有 30M 的 #1(moonshot)vs #4(keller)**不干净**(#4 混入了 lr 2.4× 变量),不能当这个实验用;要专门补一条 30M 基座对齐、lr 匹配的 keller。

2. **【关键开放问题】矩阵 wd 扫描 0.02 / 0.05 / 0.1,施加在 keller 上,跑 N7L4C128。**
   同时测三样:(a)稳定性(grad-norm 尾段、有没有跑过旧崩溃点);(b)κ_SRME;(c)force MAE 代价。
   **枢纽问题**:强 wd 在稳住的同时会不会侵蚀 keller 的 κ/一阶优势?外部团队只报了 force +15–20% 的代价,**没报 wd 对 κ 的影响**。要看 **CPS 净值**,并扫出**最小可稳定 wd**。

3. **【选型默认】"keller + 强 wd 是活跃且可能更优的候选"。**
   依据:N2L2C64(基座对齐)+ 外部 ~50M 两个独立证据都是 keller ≥ moonshot on CPS 且赢在 κ;稳定性可由矩阵 wd 单独买。待 §6.1 大模型基座对齐 A/B + §6.2 wd 扫描落地后再定稿到 `docs/MUON_PORTING_NOTES.md`。

4. **【便宜的稳健性检查】N2L2C64 keller 的 κ 优势(0.0179)是否稳?**
   vs 噪声地板 ~0.002:量级上是真的(~9× 地板),但增益小。若换 seed / 换声子子集代价低,补一个重复点确认信号稳定——这会直接影响 §4 的机制一致性解读强度。

## 7. 数据溯源

- **keller(ratio,基座对齐)弛豫**:`.../matbench/relaxation/keller_n2l2c64_gradft5ep_from_abl_moonlight15ep/{metrics.json,rmsd.txt}`(F1=0.7529 unique_prototypes,RMSD=0.07858)
- **keller(ratio,基座对齐)声子**:`.../matbench/phonon/keller_n2l2c64_gradft5ep_from_abl_moonlight15ep/`(SRME=0.4459,n_overlap=103)
- moonshot(moonlight,消融A)弛豫:`.../matbench/relaxation/abl_n2l2c64_gradft_loss-e5f10s100/{metrics.json,rmsd.txt}`(F1=0.7525,RMSD=0.07999)
- moonshot(moonlight,消融A)声子:`.../matbench/phonon/abl_n2l2c64_gradft_loss-e5f10s100/`(SRME=0.4638,n_overlap=103)
- 提交脚本:`Matbench-tools/{relaxation,phonon}/submit_keller_n2l2c64_gradft5ep_from_abl_moonlight15ep*.sh`、`submit_abl_e5f10s100.sh`
- **(留档)初版基座污染的 keller**:`.../matbench/{relaxation,phonon}/muon_ratio_n2l2c64_gradft_5ep_e5f10s100/`(2026-07-20 基座,F1=0.7844/κ=0.4718/CPS=0.7470)——**勿再用于变体归因**。
- 外部团队周报交叉比对来源:记忆 `external-mlip-muon-wd-finding`(W30, 2026-07-21)。
- CPS 汇总:`Matbench-tools/CPS_RESULTS_FULL_TABLE.md` 表1 #6b(基座对齐)vs #8(消融A)。
