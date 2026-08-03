# EquiformerV3 × MPtrj — 交接主文档 (HANDOFF)

> **这是接手者的入口文档。** 汇总:可调配置空间、已跑过的消融与结论、当前最优配方、
> 进行中/待办、以及各细分文档与文件的地图。**近期持续维护此文档直到完全交接**——
> 维护规则见文末《维护说明》。
>
> 最后更新:2026-08-03。所有测评数字的唯一真源是 [`EVAL_REGISTRY.md`](EVAL_REGISTRY.md);
> 本文档只做导航与解读,数字以 registry 的 `DONE` 行为准。

---

## 0. 目标与现状一句话

复现 MPtrj-only 的 EquiformerV3,冲 Matbench Discovery(CPS = 0.5·F1 + 0.4·max(0,1−κ_SRME/2) +
0.1·clamp((0.15−RMSD)/0.15))。**当前最优 30M 模型 CPS 0.8346(全量)/ 0.8326(5%),κ_SRME 0.2764**,
距 SOTA 约 1.7% CPS —— 因此 κ / force 上的个位数百分比提升都是显著的,不要当成 wash。

---

## 1. 环境与提交(必读,踩坑点)

| 项 | 要点 |
|---|---|
| **fairchem 版本** | 集群 docker 镜像里的 fairchem 是**旧版、没有 HybridMuon**。必须 `export PYTHONPATH="$REPO/src:$PYTHONPATH"` 指到挂载的源码,**不要重建镜像**。每个脚本开头都有一行 `assert 'HybridMuon' in inspect.getsource(base_trainer)` 自检。 |
| **Python** | 用 `./.venv/bin/python`。 |
| **代理** | 脚本开头 `unset http_proxy https_proxy all_proxy ...`(SOCKS 代理会干扰)。 |
| **提交** | SenseCore Web-UI 起 job + `SENSECORE_*` 环境变量;`/mnt/afs` 自动挂载。拓扑用 `SENSECORE_ACCELERATE_DEVICE_COUNT`(每节点 GPU)、`SENSECORE_PYTORCH_NNODES`。 |
| **推理** | 单点推理要**关 AMP**(否则 Wigner 的 fp16 index_put 崩)。 |
| **提交脚本约定** | 单行 `bash "<绝对路径>"` 启动;`torchrun --nproc_per_node=<每节点GPU> --nnodes=<节点数>`;`--num-gpus` 是**每节点** GPU 数。 |

细节见 [`docs/env.md`](env.md)、[`docs/CLUSTER_SUBMIT_WITHOUT_IMAGE_REBUILD.md`](CLUSTER_SUBMIT_WITHOUT_IMAGE_REBUILD.md)、[`docs/MUON_PORTING_NOTES.md`](MUON_PORTING_NOTES.md)。

---

## 2. 两阶段训练管线

| 阶段 | direct(stage-1) | gradft(stage-2) |
|---|---|---|
| 力的来源 | `direct_prediction: True`(力直接回归,单次反传) | `direct_prediction: False`,力 = −dE/dx(**保守力,二阶反传**) |
| 精度 | 可 **bf16**(`use_amp: True`,几何/Wigner 留 fp32 岛) | **必须 fp32**(`use_amp: False`;amp 会崩 Wigner) |
| 载入 | 从头/续训 | `optim.load_pretrained_weights=<stage-1 best_checkpoint.pt>`(非严格部分载入:backbone+energy_block 迁移,direct 力/应力头自动跳过) |
| 典型 epoch | 15 / 60 / 70ep | 5 / 10 / 40ep |

两阶段串联脚本在 `experimental/scripts/train/omat24/equiformer_v3/two_stage_pipelines/`。

---

## 3. 可调配置空间(旋钮清单)

**优化器(`optim`)**

| 旋钮 | 取值 | 说明 / 已知效应 |
|---|---|---|
| `optimizer` | `AdamW` \| `HybridMuon` | Muon 在充分训练下大幅胜 AdamW(N2L2C64 100ep:CPS 0.7773 vs 0.7317)。 |
| `optimizer_params.update_scale` | `ratio`(keller)\| `moonlight`(moonshot) | keller RMS=lr/√fan_in(形状相关、小矩阵步大、易 spike);moonshot=0.2·lr(形状解耦、跨宽度可迁移、不跨深度)。**keller κ 略优 moonshot(N2L2C64 −0.018)**。 |
| `optimizer_params.muon_lr` | 主旋钮 | moonshot gradft:30M **5e-5** / N2L2C64 **1.5e-4**;direct 更大(30M 2e-4、N2L2C64 4e-4)。keller 的 lr 用 RMS 匹配:`keller_lr ≈ 0.2·√fan_in·moonshot_lr`(如 30M 1.2e-4、N2L2C64 6.6e-4)。**过大 muon_lr 会尾部发散**(4e-4→ep9.6,2e-4→ep21;见 [[muon-divergence-lessons]])。 |
| `optimizer_params.weight_decay` | 1e-3(在用) | 外部团队发现强 wd(0.1)带来稳定,**wd=0.1 本项目尚未测**(开放项)。 |
| `optimizer_params.spike_*` | spike_factor 8、skip_nonfinite 等 | Muon 尖峰守卫:g_rms>8×EMA 跳过该矩阵、不污染动量。小矩阵(如 (8,64))触发偏多属正常,已被吸收,不发散。 |
| `lr_initial` | AdamW 组 lr(bias/norm/embed) | gradft 用 5e-5。 |

**精度 + 加速栈(maoruicong;model + optim)**

| 旋钮 | 取值 | 说明 |
|---|---|---|
| `model.use_amp` | direct True(bf16) / grad False | 块级 bf16 autocast,几何/Wigner 留 fp32。grad 必须 False。 |
| `model.enable_compile` (+`compile_dynamic`) | True/False | in-model make_fx 区域编译。**净效应待测**(见 §5 开放项)。代码层由 `if self.enable_compile` 门控,关掉即走 eager 原生路径(已代码级确认)。 |
| `optim.matmul_precision` | `highest`(纯fp32) \| `high`(TF32) \| `medium`(bf16) | 每个 torchrun 进程各自 `set_float32_matmul_precision`。**TF32-vs-fp32 在 N2L2C64 κ-中性**(旧"TF32伤κ"已推翻)。 |
| `EQV3_ACT_MEM_BUDGET`(**环境变量,不在 yml**) | [0,1] | AOTAutograd min-cut 分区的激活内存预算,**只对编译反传生效**(eager 下 inert)。1.0=不重算(快、省算)、0.6=重算(慢、省显存)。⚠️ **对 bf16 direct 非数学中性**(0.6 让 direct forces_mae 变差 −16.6%,是历史大坑,见 §4)。fp32 grad 下则 κ-中性。 |

**损失 / 数据 / 规模**

| 旋钮 | 取值 | 说明 |
|---|---|---|
| loss E:F:S | **e5f10s100**(在用) vs 论文 e20f20s5 | e5f10s100 的 κ 更好(N2L2C64:0.4638 vs 0.5208)。 |
| `max_atoms` | 150 | 对齐 30M 配方(24000 是 inert,MPtrj 最大 natoms=444);砍掉 ~0.82% 大晶胞尾。 |
| `batch_size` / `grad_accumulation_steps` | 全局恒定 512 | 横比时保持全局 512(如 30M bs8×16rank×accum4,N2L2C64 bs16×8×4)。 |
| 模型规模 N/L/C | **N2L2C64=3.5M**(廉价代理)/ **N7L4C128=30M**(生产) | 结论可能随规模衰减/反转,以 30M 为准。 |
| DPA4 开关 | `envelope_type` / `attn_softmax_type` / `focus_compete_groups` | 默认全关。**真开关(A1-D1D4)κ 灾难(1.3497)→ 已暂缓**。 |

---

## 4. 三个必须知道的历史大坑(否则会误读旧结果)

1. **budget 泄漏坑(最重要)。** 早期用顶层 `export EQV3_ACT_MEM_BUDGET=0.6`,它被 direct 阶段继承 →
   **污染了所有 bf16 direct 基座**。修正:direct 跑默认 1.0,只在 grad 的 torchrun 前内联该变量。
   凡基座是 "bf16-b0.6" 的旧 κ(A0 0.6775、E-G 0.6579、A1-D 系)**解读作废,数字保留备查**。
   → `docs/EQV3_ACT_MEM_BUDGET_direct-misapplication.md` **§4 已被推翻**(它当年说 budget 不影响 MAE/κ)。
2. **"bf16 direct 伤 κ +0.19" 已被推翻。** 分解为 budget +0.211 / TF32(现已存疑)/ bf16 本身 −0.017。
   核心是 budget bug;**bf16 做对(b1.0)后 direct 反而更好**。→ `docs/BF16_DIRECT_KAPPA_REGRESSION.md` 结论过时。
3. **"TF32/compile 伤 κ"在 N2L2C64 已被推翻,但在 30M 反转(深度放大)。** N2L2C64 四臂干净隔离
   (同 b1.0 基座):compile、TF32、grad-budget **全部 κ-中性**(跨度 0.003)。**然而**同一 maoruicong
   infra bundle(compile+HIGH+budget0.6)在 **30M 使 κ +0.0265**(旗舰 eager 0.2764 → infra-control 0.3029)。
   **教训:小模型代理会低估 infra 的 κ 代价;infra 的伤害只在深模显现。** 旗舰之所以最优,正因它 eager-fp32、
   不上 infra。(30M 尚未拆开 infra 内部是 compile 还是 TF32 还是 budget。)

**通用原则:MAE ⊥ κ。** κ_SRME 是能量面曲率(三阶力常数)的导出量,和 pointwise MAE 解耦。
精度/budget/compile 这类旋钮常常 MAE 中性但可能动 κ —— **判别力看 κ,不看 MAE**。

---

## 5. 已跑过的消融(汇总;数字详见 `docs/Matbench/`)

**结果全表**:[`docs/Matbench/N7L4C128_results.md`](Matbench/N7L4C128_results.md)(30M)、
[`docs/Matbench/N2L2C64_results.md`](Matbench/N2L2C64_results.md)(3.5M)。以下是结论速览:

| 消融维度 | 干净隔离结论 | 状态 |
|---|---|---|
| **keller vs moonshot** | keller κ 略优(N2L2C64 −0.018);lr 用 RMS 匹配 | ✅ 已定 |
| **direct budget 0.6 vs 1.0(bf16)** | 1.0 完胜(κ −0.21,forces −16.6%);0.6 是 bug | ✅ 已定 |
| **grad TF32 vs fp32** | κ-中性(N2L2C64) | ✅ 已定(翻案) |
| **grad budget 0.6 vs 0.8(fp32)** | κ-中性、MAE 中性 | ✅ 已定 |
| **grad compile vs eager(N2L2C64)** | κ-中性(eager 0.4457 vs compile 0.4471,−0.0014) | ✅ 已定 |
| **maoruicong infra bundle @30M** | **+0.0265 伤 κ**(旗舰 eager 0.2764 → infra-control 0.3029);N2L2C64 却中性=**深度放大** | ✅ 已定(关键) |
| **bf16 direct vs native** | bf16 做对后 κ、MAE 均更优 | ✅ 已定 |
| **loss e5f10s100 vs e20f20s5** | e5f10s100 更好 | ✅ 已定 |
| **DPA4 真开关** | κ 灾难(1.3497),暂缓 | ⏸️ 暂停 |
| **direct base 06-13 mlr2e-3** | MAE 最好但未传导到 κ(未来方向) | 📌 记录 |
| **30M keller(0.3086)vs 旗舰(0.2764)** | 分解 = **infra +0.0265(主导)** + 优化器 +0.0057;非 lr、非"TF32 单独" | ✅ 已定 |

**开放问题(接手者优先级):**
1. **超旗舰的干净路径**:去掉 infra 走 **eager-fp32** + **keller 优化器**(keller@30M-eager 未测;N2L2C64-eager keller −0.018)。⚠️ ~~"keller+compile+highest≈0.258"~~ **作废**:compile-infra 本身伤 κ。
2. **30M infra 内部拆分**未做(compile / TF32 / budget 各占多少)—— 需再跑,当前按用户要求不做。
3. **wd=0.1 未测**(外部发现强 wd 助稳定/助 κ)。
4. **direct base 06-13(更低 MAE)配 κ 保持型 gradft** 未系统试。

---

## 6. 当前最优配方(被超越时更新此节)

### 30M 生产模型(旗舰,当前最优)
- **ckpt**: `patched_ckpts/2026-07-09-muon_N7L4C128_gradft_10ep_from-adamw-refine_direct_bf16`
- **配方**: HybridMuon **moonshot** muon_lr 5e-5,**fp32-eager**(无 compile/TF32),10ep(best ep7),loss e5f10s100
- **direct 基座**: maoruicong **STABILIZED-70ep**(moonlight mlr2e-4,normwd1e-3)
- **成绩**: κ 0.2764,F1 **0.8696**(全量)/0.8618(5%),RMSD 0.0674/0.0646,**CPS 0.8346(全量)/0.8326(5%)**(全量 F1 为 2026-08-03 更正后的 unique_prototypes 值)
- 注:**κ 最低另有其人** —— `2026-07-03-...from-adamw-refine`(adamw-refine 基座)κ **0.2665**,但 F1/RMSD 稍逊、综合 CPS 略低(0.8283)。adamw-refine 直连线值得单独跟进。

### N2L2C64(廉价代理,注意按 epoch 分组比)
- 100ep 组最优:`muon_N2L2C64_gradft_40ep_mlr1e-3`(direct60+gradft40)κ 0.3662,CPS **0.7773**。
- 5ep 干净组内:keller-clean(0.7349)> mstack 三胞胎(0.7334–0.7344)> abl-A 基准(0.7302)。

---

## 7. 文件地图

| 类别 | 路径 |
|---|---|
| 模型代码(make_fx 版重写) | `experimental/models/equiformer_v3/`(`so3.py`、`edge_rot_mat.py` 等) |
| 训练 config | `experimental/configs/omat24/mptrj/experiments/{direct,gradient}/`(子目录:`dpa4_ablation`、`mstack_baseline`、`compile_test`) |
| 提交脚本 | `experimental/scripts/train/omat24/equiformer_v3/`(`continuation`、`mstack_baseline`、`two_stage_pipelines`、`dpa4_ablation`) |
| checkpoints | `/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/`(+ `patched_ckpts/`;30M 基座在 `.../maoruicong/checkpoints/`) |
| **测评同步(唯一真源)** | [`docs/EVAL_REGISTRY.md`](EVAL_REGISTRY.md) + 给测评侧的 prompt [`docs/EVAL_SIDE_ONBOARDING_PROMPT.md`](EVAL_SIDE_ONBOARDING_PROMPT.md) |
| **结果汇总** | [`docs/Matbench/N7L4C128_results.md`](Matbench/N7L4C128_results.md)、[`docs/Matbench/N2L2C64_results.md`](Matbench/N2L2C64_results.md) |
| 专题细分文档 | `docs/`:`ABLATION_STATUS_CONSOLIDATED`、`KELLER_VS_MOONSHOT_N2L2C64`、`N2L2C64_15plus5ep_FAMILY`、`N7L4C128_CPS_REGRESSION_ANALYSIS`、`N7L4C128_GRADFT_MEMORY_LEDGER`、`MUON_PORTING_NOTES`、`METRIC_UNITS_AND_CONVENTIONS` |
| ⚠️ 结论已过时的文档 | `EQV3_ACT_MEM_BUDGET_direct-misapplication.md`(§4)、`BF16_DIRECT_KAPPA_REGRESSION.md`(见 §4 大坑) |

**术语**:`energy_per_atom_mae`(intensive,可靠,训练目标)vs `energy_mae`(extensive,受尺寸混淆,不可靠)。
κ_SRME 恒为全量 103 结构指标,与 F1 的 5%/全量口径无关。口径细节见 `docs/METRIC_UNITS_AND_CONVENTIONS.md`。

---

## 8. 与测评侧协作(纪律)

`EVAL_REGISTRY.md` 是双方共用的唯一真源,四条纪律(不可变键=`ckpt_id`;一配方一行;列各有其主
——本项目写 recipe 列、测评侧只回填 κ/F1/RMSD;修正不动旧行)。提交训练→ append `REQUESTED` 行;
测评侧回填→ 翻 `DONE`;本项目读 `DONE` 取数。**配方整理/解读只在本项目做**。规则全文见 `CLAUDE.md`。

---

## 9. 维护说明(交接前持续更新此文档)

> **自动要求(硬性):训练/测评结果一有更新,更新本文档就是同一任务里的必做步骤,不是可选。**
> 也写进了 `CLAUDE.md`,本仓库任何 cc 都受此约束。

- **每提交一次新训练**:在 `EVAL_REGISTRY.md` append `REQUESTED` 行;在 §5 的"进行中/待回填"加一条。
- **每回填一次 κ**:更新 `docs/Matbench/*_results.md` 对应表;若结论有变,更新 §5 状态列与 §6 最优配方;
  重大翻案写进 §4 并同步 `EVAL_REGISTRY.md` 结论修正区。
- **当前最优被超越**:改 §6,旧最优移入结果汇总表(不删)。
- **开放问题解决**:从 §5 开放项移除并归档结论。
- 保持顶部"最后更新"日期与实际同步。**目标:接手者只读本文档 + `docs/Matbench/` + `EVAL_REGISTRY.md`
  三处即可完整还原现状。**
