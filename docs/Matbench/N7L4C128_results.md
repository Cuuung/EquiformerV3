# N7L4C128 (30M) — Matbench Discovery 结果

数据源:`docs/EVAL_REGISTRY.md`(唯一真源,ckpt_id 见该表)。配方经 checkpoint 实测核对
(optimizer / epochs / direct 基座 / loss / amp),精度栈与 budget 取自提交脚本。CPS 本项目按公式算出。

> **CPS** = 0.5·F1 + 0.4·max(0, 1−κ_SRME/2) + 0.1·clamp((0.15−RMSD)/0.15, 0, 1)
> **κ_SRME** 恒为全量 103 结构指标,与 F1 口径无关 → 全量表/5% 表里同一模型 κ 相同;两表差别在 F1/RMSD 弛豫口径。
> **grad 精度**:`fp32-eager`=原生(无 compile/TF32);`compile+HIGH`=maoruicong 栈+TF32;`compile+HIGHEST`=maoruicong 栈+纯 fp32。
> **grad-budget** = grad 阶段 `EQV3_ACT_MEM_BUDGET`(仅 compile 开时有效;`—`=eager,不适用)。所有 grad block 均 fp32(amp 关)。

## 全量测评(full-WBM 弛豫,cov=1.0;F1 取 **unique_prototypes.F1**,与官方 leaderboard 同口径)

> **2026-08-03 更正**:测评侧此前误填 `full_test_set.F1`,现已更正为 `unique_prototypes.F1`(旗舰 0.8522→**0.8696**、孤儿 0.8463→**0.8623**),CPS 随之上修。5% 表一直用 unique_prototypes,不受影响。

| 标签 | κ_SRME | F1(全量) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | grad ep(best) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **旗舰** | 0.2764 | 0.8696 | 0.0674 | **0.8346** | 07-14 | HybridMuon moonshot / 5e-5 | fp32-eager | — | STABILIZED-70ep(maoruicong) | 10 (7) | e5f10s100 |
| 孤儿 | 0.2851 | 0.8623 | 0.0680 | 0.8288 | 07-06 | HybridMuon moonshot / 5e-5 | fp32-eager | — | muon-direct-25ep(mlr1.5e-4) | 10 (8) | e5f10s100 |

## 5% 测评(as-predicted 弛豫,unique_prototypes)

| 标签 | κ_SRME | F1(5%) | RMSD | CPS | date | 优化器 / muon-lr | grad 精度 | grad-budget | direct 基座 | grad ep(best) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **旗舰** | 0.2764 | 0.8618 | 0.0646 | **0.8326** | 07-13 | HybridMuon moonshot / 5e-5 | fp32-eager | — | STABILIZED-70ep(maoruicong) | 10 (7) | e5f10s100 |
| 07-03 | **0.2665** | 0.8518 | 0.0665 | 0.8283 | 07-07 | HybridMuon moonshot / 5e-5 | fp32-eager | — | adamw-refine-direct-15ep | 10 (7.7) | e5f10s100 |
| infra-control | 0.3029 | 0.8609 | 0.0640 | 0.8272 | 07-31 | HybridMuon moonshot / 5e-5 | **compile+HIGH** | 0.6 | STABILIZED-70ep(maoruicong) | 10 | e5f10s100 |
| **highprec** | 0.2812 | 0.8563 | 0.0647 | 0.8288 | 08-10 | HybridMuon moonshot / 5e-5 | compile+**HIGHEST** | 0.6 | **FP32-blocks**+compile+TF32(70ep 从头) | 10 | e5f10s100 |
| keller | 0.3086 | 0.8599 | 0.0654 | 0.8246 | 07-23 | HybridMuon **keller** / 1.2e-4 | **compile+HIGH** | 0.6 | STABILIZED-70ep(maoruicong) | 10 (8) | e5f10s100 |

## 30M grad infra 分解(2026-07-31 框架,2026-08-10 拆分 TF32)

| 臂 | 优化器 | grad infra | direct | κ_SRME | 说明 |
|---|---|---|---|---|---|
| 旗舰 | moonshot | **eager-fp32** | bf16-STABILIZED(maoruicong) | 0.2764 | 最优,无 grad infra |
| **highprec** | moonshot | compile+**HIGHEST**+b0.6 | **FP32-blocks**+compile+TF32(70ep 新训) | **0.2812** | 5% CPS 0.8288(30M #2);仅丢 grad TF32 + direct 改 fp32 |
| infra-control | moonshot | compile+**HIGH**+b0.6 | bf16-STABILIZED | 0.3029 | 基准 infra bundle |
| keller | keller | compile+HIGH+b0.6 | bf16-STABILIZED | 0.3086 | infra + 换优化器 |

- **grad TF32 在 30M 单独伤 κ +0.022**(infra-control 0.3029 → highprec 0.2812;去 HIGH 上 HIGHEST + direct 改 fp32)。
- 剩余 vs 旗舰 +0.0048 = compile+budget 残留 + direct bf16→fp32 基座差(≈近旗舰,compile+b0.6 残留很小)。
- 🔴 **深度放大再确认**:同一 TF32 在 **N2L2C64 κ-中性**(0.4448 vs 0.4471),30M 却 +0.022。
- infra bundle 总伤 +0.0265 = TF32 **+0.022(主导)** + compile+budget 残留 ≈+0.005(≈旗舰附近)。

## 要点

- **旗舰 = 综合最优**(全量 CPS **0.8346** / 5% CPS 0.8326),靠 **eager-fp32 grad、无 infra**。
- **highprec = κ #2(0.2812),CPS #2(0.8288)**。direct fp32+TF32,grad compile+HIGHEST+b0.6;丢 TF32 是主因。
- 07-03(adamw-refine 基座)κ 最低 0.2665,F1/RMSD 略逊;adamw-refine 线独立跟进。
- 超旗舰:grad 需进一步**关 compile**(清 compile+budget 残留)+ 叠 keller(keller@30M-eager 未测)。
