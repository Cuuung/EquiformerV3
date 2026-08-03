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
| keller | 0.3086 | 0.8599 | 0.0654 | 0.8246 | 07-23 | HybridMuon **keller** / 1.2e-4 | **compile+HIGH** | 0.6 | STABILIZED-70ep(maoruicong) | 10 (8) | e5f10s100 |

## keller-30M 未超旗舰的分解(2026-07-31 定案,同 STABILIZED-70ep 基座)

| 臂 | 优化器 | grad infra | κ_SRME | 说明 |
|---|---|---|---|---|
| 旗舰 | moonshot | **eager-fp32** | 0.2764 | 无 infra,最优 |
| infra-control | moonshot | compile+HIGH+b0.6 | 0.3029 | 仅加 infra |
| keller | keller | compile+HIGH+b0.6 | 0.3086 | infra + 换优化器 |

- **infra 净效应 = +0.0265**(旗舰→infra-control,仅 infra 变):**maoruicong compile+HIGH+budget0.6 在 30M 伤 κ**。
- **优化器净效应 = +0.0057**(infra-control→keller):同 infra 下 keller 反略差。
- 分解:keller 总差 +0.0322 = **infra +0.0265(主导)** + 优化器 +0.0057。
- 🔴 **深度放大**:同一 infra bundle 在 **N2L2C64 κ-中性**(见 `N2L2C64_results.md` 四臂,跨度 0.003),到 30M 变 **+0.0265** → 小模型代理**低估** infra 的 κ 代价。

## 要点

- **旗舰 = 综合最优**(全量 CPS **0.8346** / 5% CPS 0.8326),靠的是 **eager-fp32、不上 maoruicong infra**。
- **07-03(adamw-refine 基座)κ 最低(0.2665)**,F1/RMSD 略逊 → CPS 略低;adamw-refine 线值得单独跟进。
- **想超旗舰应去掉 infra 走 eager-fp32,再叠 keller**(keller@30M-eager 尚未测)。~~"keller+compile+highest≈0.258"~~ **预测作废**:compile-infra 本身伤 κ,只换 highest 救不回。
- 尚未在 30M 拆开 infra 内部(compile / TF32 / budget 各自占比),需再跑,当前不做。
