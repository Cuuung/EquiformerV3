# eqV3 消融实验总账 —— 做了什么 / 结果 / 哪些干净 / 下一步

> 建档:2026-07-27 · 目的:近期为省算力把多项优化**叠在一起**跑,现发现与先前判断有差异。
> 本文把所有已跑消融、结果、以及**混杂项**理清,判定下一步哪些是必要的。
> 关联:[[N2L2C64_15plus5ep_FAMILY]]、[[BF16_DIRECT_KAPPA_REGRESSION]]、[[EQV3_ACT_MEM_BUDGET_direct-misapplication]]、[[KELLER_VS_MOONSHOT_N2L2C64]]

κ_SRME 越低越好;CPS 越高越好。

---

## 1. 已跑消融总表

| 训练 | 规模 | direct 精度 | grad 优化器/精度 | κ_SRME | CPS | 干净? |
|---|---|---|---|---|---|---|
| 消融A(moonshot) | N2L2C64 | fp16-amp(原生) | moonlight / fp32 | 0.4638 | 0.7302 | 基准 |
| keller-AB(旧) | N2L2C64 | 原生(**06-13 base**) | ratio 6.6e-4 / fp32 | 0.4718 | 0.7470 | ✗ direct base 不同 |
| 消融B | N2L2C64 | 原生 | moonlight / fp32, loss20/20/5 | 0.5208 | 0.7233 | ✓(测 loss 权重) |
| **A0-baseline** | N2L2C64 | **bf16**(budget脏) | moonlight / **TF32+compile** | 0.6775 | 0.6592 | 部分(budget/精度叠加) |
| **E-G** | N2L2C64 | bf16(=A0,budget脏) | moonlight / **fp32-eager** | 0.6579 | 未测 | ✓(隔离 grad 精度) |
| **A1-D1D4** | N2L2C64 | bf16(budget脏) | moonlight / TF32+compile + **D1+D4** | **1.3497** | 0.5203 | ✓(相对 A0 隔离 D1+D4) |
| A2-D2F2 | N2L2C64 | bf16 | + **D2(F=2)** | 待跑 | 待跑 | — |
| keller N2L2C64 kappa-AB | N2L2C64 | 原生(**06-29 moonshot base**) | ratio 6.6e-4 / fp32 | **待跑**(已配好) | — | ✓ 将是首个干净 keller vs moonshot gradft |
| **d70g10 moonshot(最佳)** | 30M | bf16 | moonlight / fp32(口述) | 0.2764 | 0.8326 | 生产最佳,超官方 |
| from-adamw-refine | 30M | **fp32** | / fp32 | 0.2665 ★最低 | 0.8282 | — |
| moonlight 06-23 | 30M | 原生(栈前) | / fp32 | 0.2851 | 0.8258 | — |
| keller 30M | 30M | bf16 | ratio 1.2e-4 / **TF32+compile** | 0.3086 | 0.8246 | ✗ 与 d70g10 grad 精度+lr 都不同 |

---

## 2. 三条线各自:已确认(干净)vs 仍混杂(需重跑)

### 线 1 — maoruicong 工程优化(bf16 direct + TF32/compile grad)

**已确认(N2L2C64,cleanish):**
- direct 精度 fp16-amp→bf16:**κ +0.19**(消融A 0.4638 ↔ E-G 0.6579,同 fp32 grad)。**bf16 是小模型 κ 主凶。**
- grad TF32+compile:**κ +0.02**(E-G 0.6579 ↔ A0 0.6775,同 bf16 direct)。**小负面(~9%)。**
- MAE(用 energy_per_atom,非 energy_mae):bf16 direct 在 N2L2C64 上 forces/E-per-atom/κ **一致更差**。

**仍混杂 / 存疑:**
- **budget bug**:A0/A1 direct 误吃 `EQV3_ACT_MEM_BUDGET=0.6` → 显存低、慢(+15%);maoruicong direct 用默认 1.0(显存高、快)。→ 我们复现不出他的显存/速度 profile。**只影响显存/速度(+κ 极小扰动),不影响 MAE/κ 主体。**(详见 budget 备忘)
- **规模依赖**:bf16 在 N2L2C64 伤 κ,在 30M 反而是最佳模型(d70g10)。**小模型结论不能外推。**
- **30M 真实 bf16 κ 代价未知**:d70g10 的 70ep 是 γ-wd 拉起来的正向,掩盖了 bf16 的 κ 代价 → #1(bf16,0.2764)vs #2(fp32,0.2665)那 0.01 被低估。
- **代码迁移**:我们 == maorui GitHub f34efc8 逐字节;但 GitHub(07-15)≠ 他实际本地(跑到 07-19,本地权限拒绝无法验)。

### 线 2 — keller vs moonshot(direct & grad)

**已确认:无一个干净对照。**
- N2L2C64(旧 keller-AB):keller CPS 0.7470 > moonshot 0.7302(赢 F1/RMSD,κ 微输)——**但 direct base 不同(06-13 vs 06-29),不可归因。**
- 30M:moonshot 0.2764 > keller 0.3086 ——**但 keller 用了 TF32 grad + 不同 lr,不可归因。**
- **方向随规模翻转**(N2L2C64 keller 胜 / 30M moonshot 胜),但两处都不干净。

**待跑(已配好,`442f330`)**:keller N2L2C64 kappa-AB(同 06-29 moonshot direct base)→ **首个干净的 keller vs moonshot gradft 单变量对照**,对 0.4638。

### 线 3 — DPA4 高阶光滑(D1/D4/D2)

**已确认:D1+D4 打包在 N2L2C64 上灾难性伤 κ。**
- A1(D1+D4)κ 1.3497,相对 A0 净 **+0.672(翻倍)**——**与"C3 截断压低 κ"的移植假设完全相反。**103 体系普遍上移,非个别爆点。
- A1−A0 是干净的(同栈,精度/budget 共模抵消)→ "D1+D4 净效应"这个结论可信。

**仍未知:**
- **D1 vs D4 无法拆**(A1 打包了两者)。D4 是核心 κ 假设却反向,最该先单独测。
- 只在 N2L2C64(小 + bf16)测过,**是否外推到 30M 未知**。
- **D2(A2)完全没测。**

**代码:D1/D2/D4 已全部移植,默认 OFF,前向 bit-identical(GATE PASS),70 项测试通过。**

---

## 3. 混杂项总清单(为什么现在和先前判断对不上)

叠加省算力,埋下 5 个混杂,逐一定位如下:

1. **budget 渗入 direct**(A0/A1/A2)→ 显存/速度失真。已定位,未修。
2. **bf16 direct**(A0/A1)→ MAE/κ 变差,但**规模依赖**(小模型伤、30M 帮)。
3. **direct base 不同**(keller-AB 旧 06-13 vs moonshot 06-29)→ keller/moonshot 不可比。已修(新 kappa-AB)。
4. **grad 精度不同**(keller 30M TF32 vs d70g10 fp32)→ 30M keller/moonshot 不可比。
5. **报告口径错标**:消融A "fp32" 实为 fp16-amp;GitHub grad-TF32 vs 口述 fp32(后者为准,脚本是事后覆盖)。

---

## 4. 下一步:必要性排序(标注算力需求)

**A. 零算力,先做(纯代码/记录):**
- [ ] 修 dpa4 流水线脚本的 budget 作用域(direct 默认 1.0,只 grad 段设 0.6)→ 防 A2/重跑重蹈。
- [x] budget 问题、消融总账已落 docs。

**B. 有算力时,按性价比:**
1. **keller N2L2C64 kappa-AB(已配好)** —— 补线 2 唯一干净对照,最便宜、直接可跑。**优先级最高。**
2. **D4 单开(只 dpa4_c3)N2L2C64** —— 线 3 核心假设。D1+D4 已灾难,必须搞清 D4 本身是无效/有害还是被 D1 拖累。零新参数、可从现成 direct 出发只跑 grad。
3. **A0 direct 默认-budget 重跑** —— 分离 budget(A线)vs bf16(B线),拿干净的显存/速度/κ。
4. **A2-D2F2** —— 补 D2 正交杠杆。
5. **(贵)30M direct-fp32 干净对照** —— 量 bf16 在生产规模的真实 κ 代价;仅当 κ 确定是冲 SOTA 的关键杠杆、且愿付 70ep 重训时才做。

**C. 需人协调(非算力):**
- [ ] 让 maoruicong push 实际本地代码 → 排除 GitHub≠本地(唯一能定"迁移是否真完整")。

---

## 5. 一句话判断

- **线 1**:bf16 direct 在小模型伤 κ(已确认),30M 上是净正向(最佳模型);budget 是显存/速度的独立 bug(可修)。
- **线 2**:keller vs moonshot **至今没有一个干净对照**,先跑已配好的 N2L2C64 kappa-AB。
- **线 3**:D1+D4 打包在小模型上**灾难性伤 κ、与假设反向**;必须**单开 D4** 定位,别急着上 30M。
- **最必要的三件**:①修 budget 脚本(零算力);②跑 keller kappa-AB(最便宜的干净对照);③单开 D4(搞清核心假设为何反向)。
