# EQV3_ACT_MEM_BUDGET 误渗入 direct 阶段 —— bug 记录 + 待办

> 建档:2026-07-27 · 状态:**已定位,未修复(暂无算力重跑)** · 关联:
> [[N2L2C64_15plus5ep_FAMILY]](N2L2C64_15plus5ep_FAMILY.md)、[[BF16_DIRECT_KAPPA_REGRESSION]](BF16_DIRECT_KAPPA_REGRESSION.md)

---

## 0. 一句话

我们的 dpa4 流水线脚本在**顶部全局 `export EQV3_ACT_MEM_BUDGET=0.6`**,而这个 env var 是**进程级全局**的编译设置——它渗进了本不该设它的 **direct 阶段**,把 direct 的显存压低、训练拖慢,使我们的 A0/A1 direct **无法复现 maoruicong 的显存/速度 profile**。这是**启动脚本的 env-var 作用域 bug,不是代码迁移缺失**(工程代码与 maorui GitHub f34efc8 逐字节一致)。

---

## 1. budget 是什么

`EQV3_ACT_MEM_BUDGET`(→ torch `functorch.config.activation_memory_budget`,范围 [0,1])是
**AOTAutograd min-cut 分区器**的旋钮,决定反向传播时"存多少前向激活 vs 重算多少":

- **1.0(默认)** = 尽量全存 → **快,显存高**(几乎不重算)。
- **0.6** = 只留一部分、其余在反向里重算 → **慢,显存低**。

**只在 `torch.compile`(inductor→AOTAutograd)编译反向时生效;eager 下无效。它不改变数学结果**
——重算出的激活与存下的是同一个值,纯粹是"显存换时间"的工程取舍。

原始用途:治 **30M 保守力 grad-ft** 的双反向 make_fx 图 saved-tensor 膨胀 / step-0 OOM。**只该用在 grad 段。**

---

## 2. bug:为什么它渗进了 direct

- 流水线脚本(`dpa4_{A0,A1,A2}_..._15ep-direct_5ep-gradft_moonshot.sh`)在**第 ~47 行顶部**
  `export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"` —— 一次导出,后续所有子进程继承。
- **STAGE-1 (DIRECT) torchrun 在其后**(~第 81 行)→ direct 子进程也拿到了 `budget=0.6`。
- direct 阶段 `enable_compile=True` → `plain_compile(core_compute, backend="inductor")`
  (`equiformer_v3.py:600-603`,`compile_utils.py:287-290`)→ inductor 走 AOTAutograd →
  min-cut 分区器读 `activation_memory_budget`(`compile_utils.py:63-71`,全局 functorch config)。
- 于是 direct 被迫重算 → **显存低 + 慢**。

**本意**是只给 grad 段;**实际**两个阶段都吃到了。env var 是全局的、不分阶段,这是根因。

---

## 3. 症状与证据(A0 direct vs 消融A direct,同 8×A100-80G / bs64 / 15ep)

| | 消融A(fp16-amp,budget 不适用) | A0(bf16 stack,budget=0.6 渗入) | maoruicong 30M direct(budget 默认 1.0) |
|---|---|---|---|
| 稳态每轮 | 1696s (28.3min) | **1946s (32.4min,+15%)** | 快(基准) |
| 总墙钟(15ep) | 7.29h | 8.24h | — |
| direct 显存 | (基准) | **≈ 消融A(低)** | **明显高于 amp**(用户实测) |

- **显存**:预期若真跑 budget 1.0 的 bf16,显存应**明显高于** amp(maoruicong 就是);我们 A0 却 ≈ amp
  → 正是 `budget=0.6` 重算把显存压回去了。
- **速度**:+15% 慢 = 重算的额外前向计算量。
- 二者叠加,精确解释"显存低 + 慢"这两个 gap。

---

## 4. ★ 关键:budget 不解释 MAE/κ 的 gap(两条独立的线)

| 线 | 变量 | 影响 | 根因 | 可否靠改 budget 修 |
|---|---|---|---|---|
| **A 线** | `budget` 0.6 vs 1.0 | 显存、速度 | 脚本 env 作用域 bug | **能**(改脚本) |
| **B 线** | 精度 bf16 vs fp16-amp | MAE、κ_SRME | bf16 尾数(7 位)比 fp16(10 位)粗,伤高阶量;**规模依赖** | **不能** |

- direct forces_mae 0.0392→0.0458(+17%)、κ 0.4638→0.6775(+46%)= **B 线(bf16)**,与 budget 无关。
- budget 重算在 bf16 下仅带来**极小**浮点顺序差(κ 的二阶扰动),**远不足以解释 +17% MAE**。
- **规模依赖**:bf16 在小模型 N2L2C64 上伤 MAE/κ(A0 更差);在 30M 上反而更好(maoruicong 版本更优)。这是 bf16 固有特性,不是 bug。

> 所以:**修 budget → direct 显存/速度追平 maoruicong;但 N2L2C64 上 bf16 的 MAE/κ 劣势不会因此消失。**

---

## 5. 受影响 / 干净 清单

**受 budget 问题影响(direct 用 budget=0.6 训的):**
- `2026-07-23-05-30-40-dpa4_A0-baseline_N2L2C64_direct_15ep_moonshot_bf16compile`(A0 direct)
- `2026-07-23-05-30-40-dpa4_A1-D1D4_N2L2C64_direct_15ep_moonshot_bf16compile`(A1 direct)
- **下游继承**:A0 gradft(`...13-45-36-dpa4_A0-baseline_...gradft`)、A1 gradft(`...07-24-04-11-44-dpa4_A1-D1D4_...gradft`)、
  E-G(`...07-24-19-48-16-EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager`,direct 基座=A0)
- **待跑**:A2-D2F2(脚本现状会重蹈)

**干净(未受影响):**
- maoruicong 30M STABILIZED direct(07-07/07-19,budget 默认 1.0)→ 及其下游 d70g10、keller 30M、from-adamw-refine。
- N2L2C64 老线 消融A / keller-AB / 消融B —— 根本没用 bf16 direct(fp16-amp/原生),无 budget、无 bf16。
- keller N2L2C64 kappa-AB(`442f330`)—— 纯 fp32 gradft、无 compile、脚本无 budget export,direct 基座是 15ep moonshot(非 bf16)。

---

## 6. 修复方案(待有算力时执行)

**代码侧(零算力,随时可做):** 把 dpa4 流水线脚本里的 budget 从"顶部全局 export"改成"**只在 grad 段 torchrun 前设**":
```bash
# 删除顶部的 export EQV3_ACT_MEM_BUDGET=...
# DIRECT torchrun 前:不设(走默认 1.0)
# GRAD-FT torchrun 前:  EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}" torchrun ...   # 只作用于这一条
```
影响脚本:`dpa4_A0-baseline_*`、`dpa4_A1-D1D4_*`、`dpa4_A2-D2F2_*`(三个流水线)。
(`KELLER-AB`/`EG`/`keller-kappaAB` 等 gradft-only 脚本无 direct 阶段,budget 只作用于 grad,**无需改**。)

**验证重跑(需算力):** 用**默认 budget** 重跑 A0 direct(其余全同),预期:
- 显存回到"明显高于 amp"、速度追平 maoruicong → 证实 A 线 = budget。
- 剩余的 MAE/κ 差 = 纯 bf16 在 N2L2C64 的代价(B 线),且不受重算污染的干净 κ。

---

## 7. 结论

- **"我们 direct 和 maoruicong 对不上"里的显存 + 速度部分 = 我脚本的 budget 作用域 bug**,已定位,非代码迁移缺失(代码 == maorui GitHub 逐字节)。
- **MAE/κ 部分 = bf16 精度(规模依赖),与 budget 无关**,修 budget 也补不回来。
- 修复分两步:改脚本(零算力,可先做)+ 默认-budget 重跑 A0 direct(待算力)。
