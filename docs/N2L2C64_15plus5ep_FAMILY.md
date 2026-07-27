# N2L2C64 15+5ep 全家谱 —— 配方 × 热导率(κ_SRME)单一索引

> 建档:2026-07-27 · 规模:N2L2C64(3.57M,2 层 64 通道)· 训练:direct 15ep + grad-ft 5ep
> 口径:Matbench-Discovery **5% as-predicted**(F1/RMSD 在 12,848 unique_prototype 上算,自比、
> 不对标 leaderboard);**κ_SRME 全量 103 声子真值**(n_overlap=103)。
> CPS = 0.5·F1 + 0.4·max(0, 1−SRME/2) + 0.1·clamp((0.15−RMSD)/0.15)。κ、RMSD 越低越好;F1、CPS 越高越好。
>
> 本表是 N2L2C64 15+5ep 家族的**单一索引**,把"配方(maoruicong 工程栈 / 优化器 / loss / DPA4 开关)"
> 与"评测(κ_SRME / CPS)"绑在一起。关联:[[KELLER_VS_MOONSHOT_N2L2C64]](KELLER_VS_MOONSHOT_N2L2C64.md)、
> [[BF16_DIRECT_KAPPA_REGRESSION]](BF16_DIRECT_KAPPA_REGRESSION.md)。

---

## 1. 全表(按 κ_SRME 升序,即热导率由好到差)

| 配方 | direct 栈 | grad 优化器 | grad lr | loss e/f/s | grad 栈 | DPA4 开关 | F1↑ | **κ_SRME↓** | RMSD↓ | **CPS↑** |
|---|---|---|---|---|---|---|---|---|---|---|
| **消融A (moonshot)** | 原生(pre-maorui) | moonlight | 1.5e-4 | 5/10/100 | fp32 | 关 | 0.7525 | **0.4638** 🥇 | 0.0800 | 0.7302 |
| **keller** | 原生(pre-maorui) | **ratio** | 6.6e-4 | 5/10/100 | fp32 | 关 | 0.7844 | 0.4718 | 0.0763 | **0.7470** 🥇 |
| **消融B** | 原生(pre-maorui) | moonlight | 1.5e-4 | **20/20/5** | fp32 | 关 | 0.7640 | 0.5208 | 0.0818 | 0.7233 |
| **E-G** | **bf16(maorui)** | moonlight | 1.5e-4 | 5/10/100 | fp32 | 关 | — | 0.6579 | — | 未测* |
| **A0-baseline** | **bf16(maorui)** | moonlight | 1.5e-4 | 5/10/100 | **TF32+compile** | 关 | 0.6995 | 0.6775 | 0.0825 | 0.6592 |
| **A1-D1D4** | bf16(maorui) | moonlight | 1.5e-4 | 5/10/100 | TF32+compile | **D1+D4** | 0.6955 | **1.3497** ❌❌ | 0.0862 | 0.5203 |
| **A2-D2F2** | bf16(maorui) | moonlight | 1.5e-4 | 5/10/100 | TF32+compile | **D2(F=2)** | — | 待测 | — | 待测 |

\*E-G 只测了 κ_SRME(见 [[BF16_DIRECT_KAPPA_REGRESSION]] §9),未跑 5% 的 F1/RMSD/CPS。

> direct 栈脚注:"原生(pre-maorui)" 指 maoruicong 混合精度栈合并之前的配方。三条 pre-maorui 线的
> direct 具体精度存在记录不一致——其 wandb config 记 `amp: true`(即整图 fp16 AMP),而
> BF16 报告 §2 标注为 "fp32 eager"。此歧义**不影响 κ 排序与本表任何结论**,待训练侧核准。

---

## 2. 每一对受控对照分别隔离了什么(表内单变量)

| 对照 | 唯一变量 | κ 变化 | 结论 |
|---|---|---|---|
| 消融A ↔ E-G | direct 精度:原生 → **bf16** | **+0.194**(0.4638→0.6579,grad 都是 fp32) | **direct bf16 是 κ 最大杠杆** |
| E-G ↔ A0 | grad 精度:fp32 → **TF32+compile** | **+0.0196**(0.6579→0.6775,direct 都是 bf16) | grad TF32 是**小**负面(~9%) |
| A0 ↔ A1 | **D1+D4 开关** | **+0.672**(0.6775→1.3497) | **D1+D4 把 κ 翻倍,与移植初衷反向** |
| 消融A ↔ 消融B | loss 权重:5/10/100 → 20/20/5 | +0.057(0.4638→0.5208) | 论文权重 20/20/5 的 κ **更差**;5/10/100 更优 |
| 消融A ↔ keller | update_scale:moonlight → ratio | +0.008(0.4638→0.4718) | κ 上 moonshot 略胜,但 keller 靠 F1/RMSD 反超 CPS |

**κ 杠杆强度排序(N2L2C64):D1+D4(+0.672)≫ direct bf16(+0.194)≫ loss 权重(+0.057)> grad TF32(+0.020)> update_scale(+0.008)。**

---

## 3. 关键结论

1. **热导率最优 = 消融A**(原生 direct + moonlight + fp32 grad + loss 5/10/100),κ_SRME 0.4638。
   **综合分最优 = keller**,CPS 0.7470(κ 微输但 F1/RMSD 赢回)。

2. **maoruicong 工程栈在这个小模型上对 κ 是净负面**:direct bf16(+0.194)+ grad TF32(+0.020)把消融A 的
   0.4638 一路推到 A0 的 0.6775。**主因是 direct bf16(≈91%),grad TF32 只占 ≈9%**(E-G 已隔离)。
   ⚠ 但这是 **N2L2C64 小模型**的结论;30M 上 bf16 的 κ 代价被规模大幅压缩(见 §4)。

3. **DPA4 的 D1+D4 在此规模上灾难性伤 κ**(+0.672,翻倍),与"C3 截断连续性改善三阶力常数、压低 κ"的
   移植假设完全相反。且 103 体系普遍上移(中位 1.434),非个别爆点。**A1 打包了 D1、D4,无法分辨谁的锅**——
   需拆单开 D4 / 单开 D1 定位。**D4 本是冲 κ 的核心假设,现在结果是反的,优先搞清。**

4. **一阶量对所有改动都钝感**:F1、RMSD 在 bf16 / TF32 / D1D4 下都基本在噪声底内不动。
   **只有 κ_SRME(二/三阶量)敏感**——凡是"改动伤 PES 曲率"的,都表现为 κ 崩、F1/RMSD 不动的指纹。

---

## 4. 与 30M 主线的关系(不要直接外推)

N2L2C64 是**快速消融平台**,但结论**多次被证明会随规模反号/衰减**:
- **bf16 direct 的 κ 代价强烈规模依赖**:N2L2C64 +0.194,30M 上则被 70ep-wd 的正向提升大幅掩盖(远小)。
  30M 上 bf16 direct 仍是净正向(使能 + 速度/显存),κ 代价的真实量级需一发干净的 30M direct-fp32 对照才能定。
- **update_scale 结论反号**:N2L2C64 是 keller>moonshot(CPS +0.0168),30M 上 moonshot>keller。
- → **小模型用于筛 loss 权重 / 精度栈方向 / DPA4 开关的定性符号;定量与最终选型以 30M 为准。**

---

## 5. ckpt(可复现)

共享盘 `/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/`:

| 配方 | gradft ckpt(被评测) |
|---|---|
| 消融A | `2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100` |
| keller | `2026-07-20-04-01-04-kappaAB_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_maxatoms150_bs16x8x4_loss-e5f10s100` |
| A0-baseline | `2026-07-23-13-45-36-dpa4_A0-baseline_N2L2C64_gradft_5ep_moonshot_compile` |
| E-G | `2026-07-24-19-48-16-EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager` |
| A1-D1D4 | `2026-07-24-04-11-44-dpa4_A1-D1D4_N2L2C64_gradft_5ep_moonshot_compile` |

(消融B / A2-D2F2 ckpt 待补;A2 gradft 就绪后接单评测,κ 补入 §1。)
