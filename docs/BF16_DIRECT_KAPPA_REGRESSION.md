# bf16+compile direct 栈导致 κ_SRME 回归 —— 受控单变量证据（交 eqV3 训练侧排查）

> 作者：Matbench 评测侧（yaolekai/MLIP/Matbench）。2026-07-24。
> 读者：equiformer_v3 训练侧 cc。
> 一句话：**同一 gradft 配方下，direct 阶段从「fp32 旧栈」换到「bf16+compile 新栈」，5% 弛豫 CPS 从 0.7302 跌到 0.6592（−0.071），重灾区是 κ_SRME +0.214、F1 −0.053。这是一个 direct 基座为唯一变量的受控对比，请训练侧定位是 bf16 / TF32 / in-model compile 三者中的哪一个（或其组合）在伤声子。**

---

## 1. 结论（TL;DR）

- 两个 N2L2C64 模型，**gradft 阶段逐字段相同**（HybridMuon moonshot / muon_lr 1.5e-4 / 5ep / loss e5:f10:s100），**direct 阶段也都是 15ep / muon_lr 4e-4 / loss e5f10s100**。
- **唯一实质差异在 direct 的数值栈**：旧的 fp32 eager vs 新的 bf16 + in-model compile + TF32 matmul。
- 结果 direct 的这点差异，经 gradft 传导后，在评测上**全面退化**，且 **κ_SRME（声子非谐热导率误差）是重灾区**：0.4638 → 0.6775（+0.214，约 +46%）。
- κ_SRME 依赖 PES 的二/三阶导（声子-声子非谐散射）。它对 direct 栈如此敏感、而 F1（一阶能量分类）退化相对小，指向 **bf16/TF32 把 PES 曲率推离了 DFT**——即"bf16 basin"假说的直接证据（此前只有 mlip-forge 侧旁证，见 §6）。

---

## 2. 受控对比（唯一变量 = direct 数值栈）

评测口径：Matbench-Discovery，**5% as-predicted**（F1/RMSD 只在 12,848 覆盖结构上算，自比口径、不对标 leaderboard；κ_SRME 是全量 103 结构真值 n_overlap=103）。CPS = 0.5·F1 + 0.4·max(0,1−SRME/2) + 0.1·clamp((0.15−RMSD)/0.15)。**同口径、同评测代码、同一天算的，可直接横比。**

| 模型 | direct 数值栈 | gradft | F1↑ | κ_SRME↓ | RMSD↓ | **CPS↑** |
|---|---|---|---|---|---|---|
| **消融A** | **fp32 eager** | moonshot mlr1.5e-4 5ep e5f10s100 | **0.7525** | **0.4638** | 0.0800 | **0.7302** |
| **A0-baseline** | **bf16 + in-model compile + TF32** | moonshot mlr1.5e-4 5ep e5f10s100 | **0.6995** | **0.6775** | 0.0825 | **0.6592** |
| Δ | — | （相同） | **−0.053** | **+0.214 ❌** | +0.003 | **−0.071** |

分量拆解（看每项对 CPS 缺口的贡献）：

| 分量 | 消融A | A0 | ΔCPS |
|---|---|---|---|
| F1 ×0.5 | 0.3763 | 0.3497 | −0.0266 |
| κ_SRMEn ×0.4（κn=1−SRME/2）| 0.4·0.7681=0.3072 | 0.4·0.6612=0.2645 | **−0.0428**（最大项）|
| RMSDn ×0.1 | 0.0467 | 0.0450 | −0.0017 |
| **CPS** | **0.7302** | **0.6592** | **−0.071** |

**60% 的 CPS 损失来自 κ_SRME**，其余来自 F1。RMSD 几乎不动。

---

## 3. ckpt 清单（可直接复现/加载）

所有路径在共享盘 `/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/`。

**消融A（fp32 旧栈）**
- direct 基座：`2026-06-29-09-38-08-abl_N2L2C64_direct_15ep_moonlight_mlr4e-4_loss-e5f10s100/best_checkpoint.pt`
- gradft（被评测）：`2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100/best_checkpoint.pt`

**A0-baseline（bf16+compile 新栈）**
- direct 基座：`2026-07-23-05-30-40-dpa4_A0-baseline_N2L2C64_direct_15ep_moonshot_bf16compile/best_checkpoint.pt`
- gradft（被评测）：`2026-07-23-13-45-36-dpa4_A0-baseline_N2L2C64_gradft_5ep_moonshot_compile/best_checkpoint.pt`

> 注：A0 是 DPA4 消融系列的 baseline 臂（三开关全 baseline 值，前向 bit-identical 于改前模型），本对比与 DPA4 三开关无关——它测的是 direct 数值栈，不是 DPA4。

---

## 4. 两个 direct 基座的完整 config 差异

逐字段 diff（`config['model']` + `config['optim']`），排除相等项：

```
model 段：
  use_amp:            消融A=None(fp32)  ->  A0=True(bf16 AMP)        ★ 嫌疑1
  matmul_precision:   消融A=None        ->  A0='high'(TF32 matmul)   ★ 嫌疑2   (在 optim 段)
  enable_compile:     消融A=None        ->  A0=True                  ┐ in-model compile
  compile_dynamic:    消融A=None        ->  A0=True                  ┘ ★ 嫌疑3
  use_compile(optim): 消融A=True        ->  A0=False                 （旧 outer compile 关，换成上面的 in-model）
  --- 以下是 DPA4 新增字段，全为 baseline 默认值，前向 bit-identical，【非】混淆项：
  envelope_type:      消融A=None -> A0='equiformerv3_c2'   (默认 C2 包络)
  attn_softmax_type:  消融A=None -> A0='equiformerv3'      (默认 softmax)
  focus_compete_groups: 消融A=None -> A0=0                 (关)
```

**关键澄清**：两个模型**都编译了**——消融A 用旧的 outer `torch.compile`（`optim.use_compile=True`），A0 用新的 in-model compile（`model.enable_compile=True` + `optim.use_compile=False`）。所以"有没有 compile"不是差异。真正变的三样：

1. **`use_amp`：fp32 → bf16**（最强嫌疑，直接改前向数值精度）
2. **`matmul_precision`：默认 → `high`**（TF32 matmul，降低乘加精度）
3. **compile 机制：outer torch.compile → in-model make_fx compile**（两条编译路径的数值可能不同）

---

## 5. 建议的拆解实验（把三个嫌疑分开）

三个变量耦合在一次 direct run 里，无法从现有 ckpt 归因。最小拆解：以 A0 的 direct 配方为基准，**每次只回退一个变量**重训 direct 15ep + 沿用同一 gradft 5ep，再送评测（评测侧脚本现成、约 1.6h 弛豫 + 1.3h 声子/条）：

| 实验 | 改动 | 预期若该项是主因 |
|---|---|---|
| **E1（最高优先）** | `use_amp=True → False`（direct 回 fp32，其余保持 bf16-A0 栈的 compile/TF32） | κ 回落到 ~0.46 → **bf16 是主因** |
| E2 | `matmul_precision='high' → 'highest'`（关 TF32） | κ 部分回落 → TF32 有份 |
| E3 | in-model compile 关（`enable_compile=False`，direct 也 eager） | κ 回落 → 编译路径数值有份 |

我方判断 **E1 最可能是主因**（bf16 直接改前向精度，且 κ 对曲率最敏感）；E1 若单独就把 κ 拉回 ~0.46，问题即定位。评测侧可随时接单跑这些 direct 变体的 5%+声子。

---

## 6. 为什么这条结论可信 + 先验证据

- **口径干净**：F1/RMSD/κ 三项均按 matbench_discovery 源码在同一评测代码里算，A0 与消融A 同一天同流程，`test_set_mode=as-predicted`、`n_evaluated=12848`、κ 的 `n_overlap=103 / n_missing=0` 均已核。
- **单变量成立**：两条 gradft 逐字段相同、direct 除数值栈外相同（含 muon_lr/epochs/loss/数据）。已逐 ckpt 核对，非名义推断。
- **先验旁证**：mlip-forge 侧的 κ 诊断（`Matbench/Matbench-tools/DIAGNOSIS_fairchem_repro_kappa.md`）早就把"direct 段 bf16 basin + PES 二三阶曲率过硬"列为 κ 灾难的残余病因之一，但此前只有跨实现的旁证；本对比是**同实现、单变量**的直接证据。
- **物理自洽**：κ_SRME 依赖三阶力常数（声子-声子散射），对 PES 曲率的敏感度远高于 F1（一阶能量分类）。观测到的"κ 重伤、F1 轻伤、RMSD 几乎不动"正是"数值精度伤曲率"的指纹。

---

## 7. 对训练侧的实际影响（为什么值得查）

- **生产配方在用 bf16 direct**：30M 的主力线 `d70g10 bf16`（full-wbm CPS 0.8346，已超官方 eqV3）direct 就是 bf16。若 bf16 单独压着 κ，说明**当前最优模型的 κ 还有肉眼可见的上行空间**——而冲 DPA4 SOTA（κ=0.211）唯一的杠杆恰恰就是 κ_SRME。把 direct 的 bf16 换回 fp32（或只对声子敏感的算子保精度）可能直接抬 CPS。
- **DPA4 消融的干扰项**：A0/A1/A2 三臂都在 bf16+compile 栈上，bf16 的退化在三臂间抵消，A1−A0/A2−A0 仍纯反映 D1/D2/D4 净效应（这条不受影响）。但**A0 的绝对值被 bf16 压低**，别拿它跨系列和 fp32 的老消融比。

---

## 附：评测侧可提供的东西

- 现成 5% 弛豫 + 声子评测脚本（旧评测镜像 equiformer_v3:mptrj，1node×4GPU/条），训练侧出任意 direct 变体 ckpt 即可接单。
- 逐结构 κ 明细（`kappa_metrics.csv`，103 条 per-material SRME），可定位是哪些声子体系被 bf16 打崩、便于缩小到具体算子。
- CPS 分量与归一脚本（`Matbench-tools/compute_cps.py`）。

---

## 8. 追加（2026-07-24）：DPA4 消融 A1-D1D4 结果 —— D1+D4 反而把 κ_SRME 翻倍

> 与前 7 节的 bf16 议题相互独立，但同一批评测里得到，一并交训练侧参考。

### 8.1 一句话

**在 A0-baseline（bf16+compile 栈）上打开 D1（`dpa4_envelope_gated`）+ D4（`dpa4_c3`），κ_SRME 从 0.6775 暴涨到 1.3497（+0.672，约翻倍），CPS 从 0.6592 跌到 0.5203（−0.139）。与「D4 的 C3 截断连续性应改善三阶力常数、压低 κ」的移植初衷完全相反。**

### 8.2 受控对比（A1 vs A0，唯一变量 = D1+D4 开关）

A0 与 A1 的 direct/gradft 数值栈逐字段相同（都是 bf16+compile direct 15ep + moonshot gradft 5ep e5f10s100），**唯一差别是三开关**：A1 置 `envelope_type=dpa4_c3` + `attn_softmax_type=dpa4_envelope_gated`（`focus_compete_groups=0` 保持关），相对 baseline +16 参数（每 block 每 head 一个 `z_bias_raw`）。

| 分量 | A0-baseline | A1-D1D4 | Δ = D1D4 净效应 |
|---|---|---|---|
| F1 | 0.6995 | 0.6955 | −0.004（噪声内）|
| κ_SRME | 0.6775 | **1.3497** | **+0.672 ❌❌** |
| RMSD | 0.0825 | 0.0862 | −0.004（噪声内）|
| **CPS** | **0.6592** | **0.5203** | **−0.139** |

**几乎全部损失来自 κ_SRME；F1/RMSD 都在噪声底内没动。** 一阶量（能量分类/结构）不受影响、二三阶量（声子）灾难性恶化——又是"改动伤 PES 曲率"的指纹（同 §6 逻辑）。

### 8.3 稳健性排查

κ=1.3497 不是少数体系爆掉拉高均值，而是**全体系普遍上移**：103 条 per-material SRME 中位数 1.434，49 条 >1.5，仅 4 条完全失败(=2.0)，仅 5 条 <0.5。即 D1D4 让**大多数声子体系的热导率预测系统性变差**。

### 8.4 无法归因到 D1 还是 D4（A1 打包了两者）

A1 同时开 D1+D4，无法分辨是哪个（或其交互）在伤 κ。若要定位，需拆成**单开 D4（只 `dpa4_c3`）**与**单开 D1（只 `dpa4_envelope_gated`）**各一臂重训。这对判断 DPA4 移植方向很关键——尤其 **D4 的 C3 截断本是冲 κ 的核心假设，现在结果是反的**，值得优先搞清是 D4 本身无效/有害，还是被 D1 拖累。

### 8.5 ckpt

- A1-D1D4 gradft（被评测，epoch 4.997）：`/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-24-04-11-44-dpa4_A1-D1D4_N2L2C64_gradft_5ep_moonshot_compile/best_checkpoint.pt`
  - direct 基座：`2026-07-23-05-30-40-dpa4_A1-D1D4_N2L2C64_direct_15ep_moonshot_bf16compile/best_checkpoint.pt`
- A0-baseline 见 §3。评测口径同 §2（5% as-predicted，κ 全量 103，n_overlap=103/n_missing=0）。
- A2-D2F2（focus-compete，正交方向）待测，gradft ckpt 就绪即接单，届时补入本表。

---

## 9. 追加（2026-07-27）：拆解实验 EG —— gradft 侧精度【不是】主因，病根锁定 direct bf16

> 回填 §5 拆解矩阵。此实验只动 gradft 侧精度、direct 的 bf16 基座保持不变，直接回答"κ 退化里
> 有多少来自 gradft 的 in-model compile + TF32，多少来自 direct 的 bf16"。

### 9.1 设置（唯一变量 = gradft 精度栈）

以 A0-baseline 为基准，**direct 基座完全不动**（仍是那份 bf16+compile 的 direct 15ep），**只把 gradft 从
「in-model compile + TF32(matmul_precision=high)」换成「FP32-eager 全精度(enable_compile=False /
use_amp=False / matmul_precision=highest)」**，gradft 其余配方不变（moonshot / mlr1.5e-4 / 5ep / e5f10s100）。
非 DPA4 臂（三开关全 baseline，132 键，0 个 DPA4 参数）。

### 9.2 结果

| | A0-baseline | EG（本次） | Δ |
|---|---|---|---|
| direct 基座 | bf16+compile | bf16+compile（同） | — |
| gradft 精度 | in-model compile + TF32 | FP32-eager 全精度 | 唯一变量 |
| **κ_SRME** | 0.6775 | **0.6579** | **−0.0196** |

SRME 分布健康（中位 0.582，46/103 体系 <0.5，仅 1 条完全失败）——模型没崩，只是被 direct bf16 压着一层降不下来的底噪。

### 9.3 结论：κ 退化几乎全在 direct bf16，gradft 精度只占 ~9%

两个单变量的代价/收益对比：

| 单变量改动 | κ 变化 | 占比 |
|---|---|---|
| direct fp32 → bf16（消融A 0.4638 → A0 0.6775） | **+0.214**（恶化） | 病根 |
| gradft compile+TF32 → FP32-eager（A0 0.6775 → EG 0.6579） | **−0.0196**（微弱回收） | ≈9% |

**清理 gradft 精度对 κ 是杯水车薪（只回收约 9%）。要救 κ 必须动 direct 的 bf16 —— §5 的 E1（direct 回 fp32）
仍是唯一有希望的解药，且现在有了量化依据：不必再在 gradft 侧折腾精度。**

> ⚠️ **后续被 §10 修正**：§10 的 mstack-clean 给出一个**名义上仍是 bf16 direct、κ 却回到好盆地**的反例，
> 说明"病根是 bf16 本身、只能靠 E1 退回 fp32"这句话下早了——真正的病根疑似是 A0/dpa4 那套**特定** direct 栈里的某个东西，
> 换成"干净"的 bf16 栈就没了。见 §10。

### 9.4 ckpt

- EG gradft（被评测，epoch 5.0）：`/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-24-19-48-16-EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager/best_checkpoint.pt`
  - direct 基座：同 A0（`2026-07-23-05-30-40-dpa4_A0-baseline_N2L2C64_direct_15ep_moonshot_bf16compile`）
- 评测口径同 §2（κ 全量 103，n_overlap=103 / n_missing=0）。

---

## 10. 追加（2026-07-27）：mstack-clean —— 一个「干净」的 bf16 direct 栈似乎逃出了 basin

> 这是本议题目前最重要的一条：它可能把 §5/§9 的"bf16 是宿命、只能退回 fp32"结论**整体推翻**。
> 但**关键变量（mstack-clean 到底改了什么）评测侧看不出来，需训练侧确认**——先把数据摆出来。

### 10.1 设置

被评测的 gradft：`mstack-clean_N2L2C64_gradft_5ep_moonshot_compile-HIGHEST-fp32`
（moonshot / muon_lr 1.5e-4 / 5ep / e5f10s100 / `enable_compile=True` / `use_amp=False` / **`matmul_precision=highest`（TF32 关）**），
非 DPA4 臂（三开关全 baseline，剔编译键后 44 键，与纯 baseline 零差异）。

它的 direct 基座是 `2026-07-27-12-41-36-mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1`：
**`use_amp=True` + `enable_compile=True`——名义上就是 bf16+compile，与 A0 的 direct 栈同一套 bf16 设置。**

### 10.2 结果:名义 bf16 direct,κ 却在好盆地

| 线 | direct 基座 | gradft 精度 | **κ_SRME** |
|---|---|---|---|
| 消融A | **fp32** | fp32 | 0.4638 |
| **mstack-clean（本次）** | **bf16compile（"clean" b1.0）** | fp32(highest+compile) | **0.4471** |
| EG | bf16compile（A0 direct） | fp32(eager) | 0.6579 |
| A0-baseline | bf16compile（A0 direct） | bf16(compile) | 0.6775 |

**mstack-clean 的 κ=0.4471 落在 fp32 好盆地,甚至略优于消融A 的 fp32 direct(0.4638)——尽管它的 direct 基座名义上是 bf16。**

### 10.3 为什么落差只能归给 direct 栈

- **gradft 精度不是原因**:§9 的 EG 已经夹死这条支路——在 A0 的 bf16 direct 基座上把 gradft 换成全 fp32,κ 只从 0.6775 回收到 0.6579（~9%）。
- 因此 mstack 从 A0 的 0.6775 跳到 0.4471 的这 **~0.23**,**不可能来自 gradft 精度,只能来自 direct 基座本身**。
- mstack 与 A0 的 direct 基座都叫 `direct_15ep_moonshot_bf16compile`、`use_amp=True`,**唯一的名义差异是 `mstack-clean` + `BUDGET1`**。

### 10.4 结论（暂定,待训练侧确认一个变量）

**bf16 本身不是 κ 的宿命病根。** 更可能是 A0/dpa4 那套**特定的** direct 栈里某个东西在伤 PES 曲率,
换成 mstack-clean 的"干净"栈后,即便仍开 bf16 autocast,κ 也回到了 fp32 档。若坐实,这比 §5 的 E1（退回 fp32）**更有价值**——
因为它保住了 bf16 的训练吞吐,同时拿回 κ。这直接关系到 30M 主力线 `d70g10 bf16`（§7）能否在不牺牲吞吐的前提下再抬 κ。

### 10.5 待澄清 / 缺口（这是能否把"basin 已修复"写死的唯一障碍）

1. **`mstack-clean` + `BUDGET1` 相对 dpa4-A0 的 direct 栈到底改了哪几行?** 两者 config 里 `use_amp`/`enable_compile` 都为 True,
   评测侧从 ckpt 看不出实质差异。需训练侧点明（是 geometry/Wigner/球谐保 fp32?compile budget 改了数值路径?换了某个 fused 算子?）。
2. **严格说非单变量**:mstack vs A0 的 direct 基座不同（都叫 bf16compile,但一个 clean 一个不 clean）。但 §10.3 已用 EG 把 gradft 支路夹死,"病根在 direct 栈"的读法成立。
3. **本实验的设计意图其实是 TF32 隔离**:被评的 `HIGHEST-fp32`（TF32 关）有个孪生 `HIGH-tf32`（TF32 开）,两者**只差 `matmul_precision` 一行**。**孪生臂尚未评**——评了才能给出 gradft 侧 TF32 的单独效应（但注意这是 gradft 侧 TF32,与 §5 的 E2=direct 侧 TF32 不是同一件事）。
4. **弛豫未完成**,暂无 F1/RMSD/CPS,本节仅 κ。

### 10.6 ckpt / config

- mstack-clean gradft（被评测,epoch 5.0,`direct_prediction=False`）:
  `.../checkpoints/2026-07-27-20-39-28-mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32/best_checkpoint.pt`
  （评测用 patched 版 `.../patched_ckpts/2026-07-27-mstack-clean_N2L2C64_gradft5ep_moonshot_HIGHEST-fp32/`,剔 6 键后 state_dict 逐字节不变）
- direct 基座:`.../checkpoints/2026-07-27-12-41-36-mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1/best_checkpoint.pt`
- 训练 config:`experimental/configs/omat24/mptrj/experiments/gradient/mstack_baseline/mstack-clean_N@2_L@2_C@64_gradft-5ep_moonshot_compile-HIGHEST-fp32.yml`（其孪生 `..._compile-HIGH-tf32.yml` 为 TF32 开臂）
- 评测口径同 §2（κ 全量 103,n_overlap=103 / n_missing=0）。
