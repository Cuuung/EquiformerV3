# N2L2C64 (3.5M) — Matbench Discovery 结果

数据源:`docs/EVAL_REGISTRY.md`(唯一真源,ckpt_id 见该表)。配方经 checkpoint 实测核对
(optimizer / epochs / direct 基座 / loss / amp),精度栈与 budget 取自提交脚本。CPS 本项目按公式算出。
N2L2C64 是廉价消融代理;现有测评**全部 5% 口径**(无全量弛豫)。**按 grad epoch 分表,不跨 ep 比较。**

> **CPS** = 0.5·F1 + 0.4·max(0, 1−κ_SRME/2) + 0.1·clamp((0.15−RMSD)/0.15, 0, 1)
> **grad 精度**:`fp32-eager`=原生;`compile+HIGH`=maoruicong 栈+TF32;`compile+HIGHEST`=maoruicong 栈+纯 fp32。
> **grad-budget** = grad 阶段 `EQV3_ACT_MEM_BUDGET`(仅 compile 开时有效;`—`=eager,不适用)。
> **direct 基座**:`native`=原生 direct;`bf16-b1.0`/`bf16-b0.6`=maoruicong bf16 direct(数字=direct 阶段 budget);`污染`=budget-bug/异常基座。

## grad 100ep(direct-60 + gradft-40,同规模最充分训练)

| 标签 | κ_SRME | F1(5%) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | grad ep(best) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **muon-100ep** | 0.3662 | 0.7994 | 0.0737 | **0.7773** | 06-18 | HybridMuon / (mlr1e-3 族) | fp32-eager | — | muon-direct-60ep(mlr1e-3) | 40 (23) | e5f10s100 |
| AdamW-100ep | 0.4718 | 0.7570 | 0.0786 | 0.7317 | 06-30 | AdamW / 5e-5 | fp32-eager | — | AdamW-direct-60ep | 40 | e5f10s100 |

## grad 5ep(消融主战场)— 按 CPS 降序

| 标签 | κ_SRME | F1(5%) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| kappaAB-old | 0.4718 | 0.7844 | 0.0763 | **0.7470** | 07-21 | HybridMuon **keller** / 6.6e-4 | fp32-eager | — | **污染**(06-13 mlr2e-3) | e5f10s100 |
| **keller-clean** | 0.4459 | 0.7529 | 0.0786 | 0.7349 | 07-28 | HybridMuon **keller** / 6.6e-4 | fp32-eager | — | native | e5f10s100 |
| mstack-HIGH-tf32 | 0.4448 | 0.7496 | 0.0772 | 0.7344 | 07-29 | HybridMuon moonshot / 1.5e-4 | **compile+HIGH** | 0.6 | bf16-b1.0 | e5f10s100 |
| mstack-HIGHEST | 0.4471 | 0.7488 | 0.0771 | 0.7336 | 07-28 | HybridMuon moonshot / 1.5e-4 | **compile+HIGHEST** | 0.6 | bf16-b1.0 | e5f10s100 |
| mstack-BUDGET0.8 | 0.4477 | 0.7493 | 0.0776 | 0.7334 | 07-29 | HybridMuon moonshot / 1.5e-4 | compile+HIGHEST | **0.8** | bf16-b1.0 | e5f10s100 |
| abl-A(基准) | 0.4638 | 0.7525 | 0.0800 | 0.7302 | 06-30 | HybridMuon moonshot / 1.5e-4 | fp32-eager | — | native | e5f10s100 |
| abl-lossB | 0.5208 | 0.7640 | 0.0818 | 0.7233 | 06-30 | HybridMuon moonshot / 1.5e-4 | fp32-eager | — | native | **e20f20s5** |
| dpa4-A0 | 0.6775 | 0.6995 | 0.0825 | 0.6593 | 07-24 | HybridMuon moonshot / 1.5e-4 | **compile+HIGH** | 0.6 | bf16-b0.6 | e5f10s100 |
| dpa4-A1D1D4 | 1.3497 | 0.6955 | 0.0862 | 0.5203 | 07-24 | moonshot + **DPA4 开关 ON** | compile+HIGH | 0.6 | bf16-b0.6 | e5f10s100 |
| AdamW-5ep(仅声子) | 0.6062 | — | — | — | 07-08 | AdamW / 5e-5 | fp32-eager | — | AdamW-direct-15ep | e5f10s100 |
| E-G(仅声子) | 0.6579 | — | — | — | — | HybridMuon moonshot / 1.5e-4 | fp32-eager | — | bf16-b0.6 | e5f10s100 |

## mstack 三胞胎:精度 × budget 干净隔离(同 bf16-b1.0 基座 / compile 恒开 / 同优化器 / 5ep)

| 对照 | 变量 | κ_SRME | Δκ | 读数 |
|---|---|---|---|---|
| HIGHEST-b0.6(基准) | — | 0.4471 | — | 基准 |
| HIGH-tf32-b0.6 | HIGHEST→**HIGH(TF32)** | 0.4448 | **−0.0023** | TF32 **不伤 κ**(略降,噪声内) |
| HIGHEST-b0.8 | budget 0.6→**0.8** | 0.4477 | +0.0006 | fp32 下 budget **κ-中性** |

- 三者 κ 跨度仅 0.003(κ 噪声级),MAE 也几乎重合 → **在 N2L2C64,TF32-vs-fp32、budget-0.6-vs-0.8 均 κ-中性**。
- ⚠️ **推翻旧结论**:此前把 A0(compile+HIGH) vs E-G(eager+HIGHEST) 的 +0.020 记为"TF32 伤 κ",但那对**把 compile 和精度绑在一起**了。本次干净隔离(compile 恒开、仅动精度)显示 TF32 无害 → 那 +0.020 应归 **compile(eager↔编译)**,不是 TF32。详见 `EVAL_REGISTRY.md` 结论修正区。

## 要点

- **100ep 组**(单独可比):muon(CPS 0.7773)远优于 AdamW(0.7317)—— Muon 优化器在充分训练下大幅领先。
- **5ep 组**(消融主战场):干净对比 **keller-clean 0.7349 > mstack 三胞胎(0.7334–0.7344)> abl-A 基准 0.7302** —— keller 优化器、bf16-做对(b1.0)均优于 moonshot-native 基准;精度/budget 在组内近中性。
  - kappaAB-old CPS 最高(0.7470)但**基座污染**(06-13 mlr2e-3),不与干净组同列可比。
- **budget=0.6 的锅在 direct(bf16),不在 grad**:dpa4-A0(bf16-b0.6 direct)CPS 骤降 0.6593;bf16 direct 做对(b1.0)后 mstack 三胞胎全部反超基准。grad 阶段 budget(fp32)则 κ-中性。
- **TF32 在 N2L2C64 无害** → 30M keller(κ0.3086)未超旗舰的归因需重审:要么 TF32 的代价只在 30M 深度显现,要么另有其因(compile / 基座 / keller@30M)。
- **DPA4 真开关(A1-D1D4)κ 灾难 1.3497**,已暂缓。
