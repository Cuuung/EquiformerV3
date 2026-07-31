# N7L4C128 (30M) — Matbench Discovery 结果

数据源:`docs/EVAL_REGISTRY.md`(唯一真源,ckpt_id 见该表)。配方经 checkpoint 实测核对
(optimizer / epochs / direct 基座 / loss / amp),精度栈与 budget 取自提交脚本。CPS 本项目按公式算出。

> **CPS** = 0.5·F1 + 0.4·max(0, 1−κ_SRME/2) + 0.1·clamp((0.15−RMSD)/0.15, 0, 1)
> **κ_SRME** 恒为全量 103 结构指标,与 F1 口径无关 → 全量表/5% 表里同一模型 κ 相同;两表差别在 F1/RMSD 弛豫口径。
> **grad 精度**:`fp32-eager`=原生(无 compile/TF32);`compile+HIGH`=maoruicong 栈+TF32;`compile+HIGHEST`=maoruicong 栈+纯 fp32。
> **grad-budget** = grad 阶段 `EQV3_ACT_MEM_BUDGET`(仅 compile 开时有效;`—`=eager,不适用)。所有 grad block 均 fp32(amp 关)。

## 全量测评(full-WBM 弛豫,cov=1.0,leaderboard 口径)

| 标签 | κ_SRME | F1(全量) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | grad ep(best) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **旗舰** | 0.2764 | 0.8522 | 0.0674 | **0.8259** | 07-14 | HybridMuon moonshot / 5e-5 | fp32-eager | — | STABILIZED-70ep(maoruicong) | 10 (7) | e5f10s100 |
| 孤儿 | 0.2851 | 0.8463 | 0.0680 | 0.8208 | 07-06 | HybridMuon moonshot / 5e-5 | fp32-eager | — | muon-direct-25ep(mlr1.5e-4) | 10 (8) | e5f10s100 |

## 5% 测评(as-predicted 弛豫,unique_prototypes)

| 标签 | κ_SRME | F1(5%) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | grad ep(best) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **旗舰** | 0.2764 | 0.8618 | 0.0646 | **0.8326** | 07-13 | HybridMuon moonshot / 5e-5 | fp32-eager | — | STABILIZED-70ep(maoruicong) | 10 (7) | e5f10s100 |
| 07-03 | **0.2665** | 0.8518 | 0.0665 | 0.8283 | 07-07 | HybridMuon moonshot / 5e-5 | fp32-eager | — | adamw-refine-direct-15ep | 10 (7.7) | e5f10s100 |
| keller | 0.3086 | 0.8599 | 0.0654 | 0.8246 | 07-23 | HybridMuon **keller** / 1.2e-4 | **compile+HIGH** | 0.6 | STABILIZED-70ep(maoruicong) | 10 (8) | e5f10s100 |

## 要点

- **旗舰 = 综合最优**(全量 CPS 0.8259 / 5% CPS 0.8326)。
- **07-03(adamw-refine 基座)κ 最低(0.2665 < 旗舰 0.2764)**,但 F1/RMSD 略逊 → 综合 CPS 稍低;唯一走 adamw-refine 线,κ 方向值得单独跟进。
- **keller 30M(κ 0.3086)未超旗舰**,唯一变量是 grad 用了 **TF32(HIGH)**;小模型消融证明 keller 优化器本身优于 moonshot(−0.018 κ)。预测 keller 换 **HIGHEST** ≈ 0.258 可超旗舰(见 `EVAL_REGISTRY.md` 结论修正区)。
