# EVAL_REGISTRY — 训练侧 ↔ 测评(Matbench)侧 同步登记表

**唯一真源 (single source of truth)。** 两侧只读写这一个文件。git diff 即同步日志。
测评侧**不做任何判断**——只把结果按不可变标识回填进对应行。

---

## 四条纪律

1. **不可变键 = `ckpt_id`**(我们提交时设的 `--identifier` 字符串,也是 ckpt 目录的后缀)。
   一旦写入**永不改名、永不删行**。时间戳前缀是运行时才有的,不算键;补进 `ckpt_path` 列即可。

2. **一配方一行。** 换配方 / 重训 = 新 ckpt = **新的一行**。绝不覆盖旧行的键或旧结果。

3. **列各有其主。** 训练侧只写 `status / ckpt_id / ckpt_path / stage / recipe / notes`;
   测评侧只写 `κ_SRME / F1 / RMSD / CPS / eval_date`,并把 `status` 从 `REQUESTED` 翻成 `DONE`。
   训练侧**不填**结果列(留 `—`);测评侧**不动**配方列。

4. **结论变更 / 旧配方修正,不动旧行。** 旧 ckpt 的旧数字保持原样,
   在 `notes` 里加一句解释,或新增一行并写 `superseded-by: <ckpt_id>` 指向后继。
   —— 保证测评侧永远能用它手里那个键找回同一行。

**流转:** 训练侧提交时 append 一行 `REQUESTED`(结果列留空)→ 测评侧扫 `REQUESTED` 行、跑完回填、翻 `DONE`。
测评侧只需盯 `REQUESTED`;训练侧只需盯 `DONE` 取数。无需任何口头对齐。

> CPS = 0.5·F1 + 0.4·max(0, 1−SRME/2) + 0.1·clamp((0.15−RMSD)/0.15)。测评侧给原始 κ/F1/RMSD 即可,CPS 可留空由训练侧算。

---

## 登记表

`status`: `REQUESTED`(待测) / `DONE`(已回填) / `SUPERSEDED`(结论被后继行取代,数字仍保留)

| status | ckpt_id (不可变键) | ckpt_path (时间戳目录, 运行后补) | stage | recipe (简) | κ_SRME | F1 | RMSD | CPS | eval_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DONE | muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5 | 2026-07-09-muon_N7L4C128_gradft_10ep_from-adamw-refine_direct_bf16 (patched_ckpts;真实d70g10,maoruicong训,原始训练目录在其名下) | grad 30M | moonlight, fp32-eager, dir=stabilized70ep-bf16-b1 | 0.2764 | 0.8618 (5%) | 0.0646 | — | 2026-07-13 | 当前最优 30M |
| DONE | muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5 | 2026-07-09-muon_N7L4C128_gradft_10ep_from-adamw-refine_direct_bf16 (patched_ckpts;真实d70g10,maoruicong训,原始训练目录在其名下) | grad 30M | moonlight, fp32-eager, dir=stabilized70ep-bf16-b1 | 0.2764 | 0.8696 (全量) | 0.0674 | — | 2026-07-14 | 当前最优 30M;全量 CPS=0.8346(F1 取 unique_prototypes,超官方+0.0049);2026-07-31 修正:曾误填 full_test_set 0.8522 |
| DONE | keller_N7L4C128_gradft_10ep_RATIO_mlr1.2e-4_KAPPA-AB | 2026-07-21-09-31-44-keller_N7L4C128_gradft_10ep_RATIO_mlr1.2e-4_from-STABILIZED70ep-bf16_KAPPA-AB | grad 30M | keller ratio, compile+**HIGH(TF32)**, dir=70ep | 0.3086 | 0.8599 (5%) | 0.0654 | — | 2026-07-23 | 未超 0.2764;归因=grad TF32 被深模放大(见 notes) |
| DONE | abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4 | 2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100 | grad 3.5M | moonshot, fp32-eager, dir=native | 0.4638 | 0.7525 (5%) | 0.0800 | — | 2026-06-30 | N2L2C64 基准 |
| DONE | keller_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_KAPPA-AB | 2026-07-27-07-21-36-keller_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_from-abl-moonlight15ep-direct_KAPPA-AB | grad 3.5M | keller ratio, fp32-eager, dir=native | 0.4459 | 0.7529 (5%) | 0.0786 | — | 2026-07-28 | keller **优于** moonshot −0.018(同 direct 基座) |
| DONE | mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32 | 2026-07-27-20-39-28-mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32 | grad 3.5M | moonshot, compile+**HIGHEST(fp32)**, dir=bf16-b1.0 | 0.4471 | 0.7488 (5%) | 0.0771 | — | 2026-07-28 | bf16 做对(budget=1)后 **优于** native |
| DONE | EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager | 2026-07-24-19-48-16-EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager | grad 3.5M | moonshot, fp32-eager, dir=bf16-**b0.6** | 0.6579 | — | — | — | — | budget-0.6 污染的 direct 基座 |
| DONE | dpa4_A0-baseline_N2L2C64_gradft_5ep_moonshot_compile | 2026-07-23-13-45-36-dpa4_A0-baseline_N2L2C64_gradft_5ep_moonshot_compile | grad 3.5M | moonshot, compile+**HIGH(TF32)**, dir=bf16-b0.6 | 0.6775 | 0.6995 (5%) | 0.0825 | — | 2026-07-24 | budget-0.6 + TF32 双重污染 |
| DONE | mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGH-tf32 | 2026-07-28-04-54-24-mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGH-tf32 | grad 3.5M | moonshot, compile+**HIGH(TF32)**, budget0.6, dir=bf16-b1.0 | 0.4448 | 0.7496 (5%) | 0.0772 | — | 2026-07-29 | TF32 纯净消融,对照 HIGHEST-fp32(0.4471) |
| DONE | mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32_BUDGET0.8 | 2026-07-28-04-54-24-mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32_BUDGET0.8 | grad 3.5M | moonshot, compile+HIGHEST(fp32), **budget0.8**, dir=bf16-b1.0 | 0.4477 | 0.7493 (5%) | 0.0776 | — | 2026-07-29 | grad budget 0.6→0.8 消融,对照 budget0.6(0.4471) |
| DONE | mptrj_grad-ft_N@2_L@2_C@64_5ep | 2026-06-06-16-27-44-mptrj_grad-ft_N@2_L@2_C@64_5ep | grad 3.5M | AdamW native gradft 5ep(*配方待训练侧核*) | 0.6062 | — | — | — | 2026-07-08 | ★测评侧登记(键/配方待训练侧核);仅声子 |
| DONE | mptrj_grad-ft_N@2_L@2_C@64_40ep | 2026-06-08-02-10-08-mptrj_grad-ft_N@2_L@2_C@64_40ep | grad 3.5M | AdamW 优化器对照 direct60+gradft40(*待核*) | 0.4718 | 0.7570 (5%) | 0.0786 | — | 2026-06-30 | ★测评侧登记(待核);AdamW baseline,旧目录名 esen(esen 与 adamw 同 ckpt 已合并本行);声子06-17/弛豫06-30 |
| DONE | muon_N2L2C64_gradft_40ep_mlr1e-3 | 2026-06-16-18-57-04-muon_N2L2C64_gradft_40ep_mlr1e-3 | grad 3.5M | Muon mlr1e-3, direct60+gradft40=100ep(*待核*) | 0.3662 | 0.7994 (5%) | 0.0737 | — | 2026-06-18 | ★测评侧登记(待核);同规模最充分训练 |
| DONE | abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_loss-e20f20s5 | 2026-06-29-16-25-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e20f20s5 | grad 3.5M | moonlight, loss e20f20s5(*待核*) | 0.5208 | 0.7640 (5%) | 0.0818 | — | 2026-06-30 | ★测评侧登记(待核);loss 权重消融 B,对照上表消融 A(e5f10s100) |
| DONE | muon_N7L4C128_gradft_from-adamw-refine | patched_ckpts/2026-07-03-muon_N7L4C128_gradft_from-adamw-refine | grad 30M | Muon moonlight, from-adamw-refine(*待核*) | 0.2665 | 0.8518 (5%) | 0.0665 | — | 2026-07-07 | ★测评侧登记(待核);30M,≠旗舰(旗舰=07-09 direct_bf16 那条) |
| DONE | kappaAB_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_loss-e5f10s100 | 2026-07-20-04-01-04-kappaAB_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_maxatoms150_bs16x8x4_loss-e5f10s100 | grad 3.5M | keller ratio mlr6.6e-4, loss e5f10s100(*待核*) | 0.4718 | 0.7844 (5%) | 0.0763 | — | 2026-07-21 | ★测评侧登记(待核);旧 ratio,**污染基座**(≠上表干净 keller N2L2C64 κ0.4459) |
| DONE | dpa4_A1-D1D4_N2L2C64_gradft_5ep_moonshot | patched_ckpts/2026-07-24-dpa4_A1-D1D4_N2L2C64_gradft_5ep_moonshot | grad 3.5M | moonshot, DPA4 A1-D1D4 真开关(*待核*) | 1.3497 | 0.6955 (5%) | 0.0862 | — | 2026-07-24 | ★测评侧登记(待核);DPA4 真开关臂(focus_compete_groups 生效),κ 灾难,对照 dpa4_A0 |
| DONE | mstack-clean_N2L2C64_gradft_5ep_moonshot_EAGER-highest-fp32 | 2026-07-29-05-39-12-mstack-clean_N2L2C64_gradft_5ep_moonshot_EAGER-highest-fp32 | grad 3.5M | moonshot, **EAGER**(compile OFF)+HIGHEST(fp32), dir=bf16-b1.0 | 0.4457 | 0.7493 (5%) | 0.0776 | — | 2026-07-31 | compile-vs-eager 净效应,对照 mstack HIGHEST(compile,0.4471)→ **compile 在 N2L2C64 κ-中性**(−0.0014) |
| DONE | moonshot-infra-control_N7L4C128_gradft_10ep_moonshot_mlr5e-5_compile-HIGH-b0.6_from-STABILIZED70ep | 2026-07-29-05-47-44-moonshot-infra-control_N7L4C128_gradft_10ep_moonshot_mlr5e-5_compile-HIGH-b0.6_from-STABILIZED70ep | grad 30M | moonshot + maoruicong infra(compile+HIGH(TF32)+budget0.6), dir=STABILIZED-70ep | 0.3029 | 0.8609 (5%) | 0.0640 | — | 2026-07-31 | infra 净效应:vs 旗舰(moonshot eager)**+0.0265**、vs keller(同 infra)+0.0057 → **infra bundle 在 30M 伤 κ**(N2L2C64 却中性=深度放大) |

---

## 结论修正区(不动上表旧行,只在此追加"当前如何解读")

- **【2026-07-29 修正】"TF32 伤 κ" 已被推翻(干净三胞胎)。** 此前用 A0(compile+HIGH) vs
  E-G(**eager**+HIGHEST) 得出 "TF32 +0.020",但那对把 **compile 与精度绑在一起**。同 bf16-b1.0 基座、
  **compile 恒开**、budget0.6、moonshot、5ep,仅动 matmul 的干净隔离:
  HIGHEST-fp32 **0.4471** / HIGH-TF32 **0.4448**(−0.0023)/ BUDGET0.8 **0.4477**(+0.0006)。
  → 在 N2L2C64,**TF32 与 grad-budget 均 κ-中性**(跨度 0.003=噪声);A0/E-G 的 +0.020 应归 **compile**,非 TF32。
- **【2026-07-31 定案】`keller_N7L4C128 ... HIGH`(κ0.3086)未超旗舰 = maoruicong infra bundle 伤 κ(深度放大)。**
  两条 30M 干净隔离(同 STABILIZED-70ep 基座)拿到:
  - **infra 净效应**(旗舰 moonshot-eager 0.2764 → infra-control moonshot-compile+HIGH+b0.6 **0.3029**)= **+0.0265,infra bundle 伤 κ**。
  - **优化器净效应**(infra-control 0.3029 → keller 同 infra 0.3086)= +0.0057(keller 在此 infra 下反略差)。
  - 分解:keller-30M 总差 +0.0322 = infra **+0.0265(主导)** + 优化器 +0.0057。
  **关键**:同一 infra bundle 在 N2L2C64 κ-中性(compile/TF32/budget 三胞胎跨度 0.003),但在 30M 深模 +0.0265
  → **深度放大,小模型代理低估 infra 的 κ 代价**。**"keller+compile+highest≈0.258" 预测作废**(compile-infra 本身伤 κ,
  只换 highest 救不回)。→ 想超旗舰应**去掉 infra 走 eager-fp32**,再叠 keller 优化器(keller@30M-eager 尚未测)。
  尚未在 30M 拆开 infra 内部(compile / TF32 / budget 各自占比)—— 若要拆需再跑,当前不做。
- **"bf16 direct 使 κ +0.19" 的旧结论已被推翻:** 分解为 budget +0.211 / TF32(现存疑,见上)/ bf16 本身 −0.017。
  核心是 **budget=0.6 的 bug**(direct 阶段,bf16 下非中性);bf16 本身反而降 κ。凡基于 budget-0.6 direct 的旧 κ(A0/E-G 及 A1-D 系)解读作废,数字保留备查。

---

## 待训练侧登记(测评侧填,键未知)

测评侧若有结果但上表无匹配 `ckpt_id`,**不要造行/猜键**,在此按结果时间顺序追加一条,
带上任何能识别该模型的信息 + 数字 + 日期。训练侧会补上正确 `ckpt_id` 后再回填上表。

- *(示例)* `<ckpt路径 / 时间戳 / 收到的名字>` → κ=…, F1=…, RMSD=…, date=YYYY-MM-DD

> 口径说明(测评侧填):`F1 (5%)` = 5% as-predicted 的 `unique_prototypes.F1`;`F1 (全量)` = 全量 full-wbm(cov=1.0)弛豫的 **`unique_prototypes.F1`**(与官方/DPA4 同口径、可直接对标 leaderboard;⚠️ **不是** `full_test_set.F1`——5% 与全量一律取 unique_prototypes.F1,只是覆盖数不同);`κ_SRME` = 全量 103 结构 `srme_mean`(均 n_overlap=103);`RMSD` = `structure_rmsd_vs_dft`;`date` = 测评结果产生日(metrics.json mtime)。`F1=—/RMSD=—` 表示只跑了声子、未跑弛豫。一个训练若 5%/全量弛豫都有,则登两条、仅 F1 区分。**只登原生 EquiformerV3;mlip-forge/fairchem2 家族不登。** 标识用我这边的**测评输出目录名**(粗体)+ 已知 gradft ckpt 时间戳目录,训练侧据此配 `ckpt_id`。

<!-- 2026-07-29:历史孤儿已按指示全部升入上表(esen+adamw 同 ckpt 已合并)。此处仅剩 1 条 key 冲突待训练侧裁决;mlip-forge 家族按指示不登。 -->

- 2026-07-06 · **`muon_n7l4c128_gradft_10ep_moonlight_mlr5e-5`**(2026-06-23-04-09-36 纯 moonlight ckpt)→ κ=0.2851, F1=0.8623 (全量), RMSD=0.0680, date=2026-07-06 · ⚠️ **key 冲突,未敢升表**:其自然键 `muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5` 与上表旗舰行的不可变键**完全同名**,但旗舰行实为 d70g10(已把 ckpt_path 更正为 `2026-07-09-...from-adamw-refine_direct_bf16`)。此为 2026-06-23 独立 moonlight 训练(κ=0.2851,≠旗舰 0.2764)。其 5% 弛豫误用 full-wbm 跑、F1 污染不可用,故只登全量弛豫 F1。**请训练侧裁决键名后我再升表**
- ~~2026-07-31 · `mstack_clean_n2l2c64_gradft5ep_eager_highest_fp32`~~ **→ 已升表 2026-07-31**(ckpt_id=`mstack-clean_N2L2C64_gradft_5ep_moonshot_EAGER-highest-fp32`,κ=0.4457)。
- ~~2026-07-31 · `moonshot_infra_ctrl_n7l4c128_gradft10ep_high_b06`~~ **→ 已升表 2026-07-31**(ckpt_id=`moonshot-infra-control_N7L4C128_gradft_10ep_moonshot_mlr5e-5_compile-HIGH-b0.6_from-STABILIZED70ep`,κ=0.3029)。
