# N7L4C128(30M)当前跑通版 vs 官方配方 —— 差异对比

> 对比对象:
> - **官方**:仓库随附的 AdamW 配方(`Init` 提交即存在,注释标明 follow paper Table 10 + released checkpoint)
>   - direct:`experiments/direct/equiformer_v3_N@7_L@4_..._epochs@70-bs@512-wd@1e-3-beta2@0.95_dens-...-no-stress.yml`
>   - grad-ft:`experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_..._lr@0-5e-5-epochs@10-bs@64x8-....yml`
> - **当前跑通版**:HybridMuon 配方
>   - direct:`experiments/direct/equiformer_v3_N@7_L@4_C@128_..._epochs@25-bs@32x16_hybridmuon-moonlight-mlr@1.5e-4-...-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml`
>   - grad-ft:`experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_..._hybridmuon-moonlight-mlr@5e-5-...-epochs@10-...loss-e5-f10-s100.yml`
>
> 背景:本仓库即官方 EquiformerV3 实现,`equiformer_v3_dens` 骨架沿用官方。以下所有差异都在**训练配方 / 优化器 / 精度 / 数据侧**,**不含任何模型架构改动**。

## 模型架构:零差异

`model:` 块逐字段一致,确认未改任何结构:

```
num_layers=7   num_channels=128   lmax=4   mmax=2
attn_hidden=32   num_heads=8   attn_alpha=64   attn_value=16   ffn_hidden=512
attn_grid=[14,8]   ffn_grid=[14,14]   edge_channels=128   norm=merge_layer_norm
attn/ffn_activation=sep-merge_gates2_swiglu(direct)   use_gate_force_head=True
use_rad_l_parametrization=True   use_grid_mlp=True   use_envelope=True
drop_path=0.05 / attn_weights_drop=0.1(direct)   avg_num_nodes=1
gradient_checkpointing_block_list=[0]*7   max_neighbors=300   max_radius=6.0   num_radial_basis=10
```

## 差异对比表

| # | 类别 | 字段 | 官方(AdamW 配方) | 当前跑通版(HybridMuon) | 阶段 |
|---|---|---|---|---|---|
| 1 | **优化器** | `optimizer` | AdamW(betas 0.9/0.95) | **HybridMuon**(Muon 管 ≥2D 矩阵 + AdamW 管 bias/norm/embed);`update_scale=moonlight` | 两阶段 |
| 2 | 优化器 | `muon_lr` | —(无) | direct **1.5e-4** / gradft **5e-5**;`momentum=0.95 / nesterov / ns_steps=5` | 两阶段 |
| 3 | **损失权重** | E:F:S 系数 | **20:20:5**(paper + checkpoint) | **5:10:100** | 两阶段 |
| 4 | **训练时长** | `max_epochs`(direct) | **70** | **25**(Muon 收敛快,ep20 已 cos 0.756;前载 + 早衰以躲尾段 runaway) | direct |
| 5 | **发散保护** | spike 守卫 | —(AdamW 不需要) | `skip_nonfinite` + `spike_factor=8` + `spike_max_consecutive_skips=25` + **`spike_abs_threshold=4.0`** | 两阶段 |
| 6 | 调度 | `warmup_epochs` / `warmup_factor` | 0.1 / 0.0 | **0.5 / 1e-3**(更长更缓,喂 Muon 进满 lr) | direct |
| 7 | 调度 | `lr_min_factor` | 0.01 | **0.001**(更陡尾段,末端步长压小躲 runaway) | direct |
| 8 | **数据过滤** | `max_atoms`(gradft) | **24000**(失效:MPtrj 最大才 444 原子) | **150**(grad-ft OOM 根因修复,丢 0.82% 尾巴) | gradft |
| 9 | **算子选择** | `attn_activation`(gradft) | `sep-merge_gates2_swiglu_mem`(省显存变体) | `sep-merge_gates2_swiglu`(与 direct 一致,保证权重迁移) | gradft |
| 10 | **验证集** | `val.src` | sAlex `val_30k` | **MPtrj `aselmdb/val`** | 两阶段 |
| 11 | (次要)吞吐 | 每卡 batch 布局 | 64/卡(global 512) | 32/卡 × 16 = 512(global 相同,仅卡数布局不同) | direct |

**未变项(易误认为差异,实为一致)**:`lr_initial`(非矩阵 AdamW lr,direct 2e-4 / gradft 5e-5)、`clip_grad_norm=100`、`ema_decay=0.999`、gradft 的 `batch=8 × accum=4`、`direct_prediction`(direct True / gradft False = conservative 力)、`use_compile`(direct True / gradft False)、gradft dropout 全关、gradft fp32 / no-amp —— 两边相同。

## 状态说明

- **第 4–7 项**是绕开"深模型 Muon 尾段静默 runaway"的组合拳。`direct 25ep` 是**已确认健康的首个 30M 全程**(val cos **0.7516** / forces_mae 0.0273 / loss 0.9205 @ep25;ep15–22 危险带干净,abs backstop 从未触发)。
- **grad-ft 阶段**:第 8/9 项是 OOM 修复,config 已落地(`bs=8 × accum=4 + max_atoms=150`);完整收敛的健康 gradft run 需核对 checkpoint 指标后确认。
- **第 3 项(5:10:100 vs 官方 20:20:5)**正在 N2L2C64 上做单变量消融验证;若结论支持 20:20:5,30M 版应把该项也对齐官方。

## 相关文档 / memory
- `MUON_PORTING_NOTES.md`(HybridMuon 移植)、`CLUSTER_SUBMIT_WITHOUT_IMAGE_REBUILD.md`(集群提交)
- memory:`muon-migration-retrospective`、`muon-divergence-lessons`、`adamw-vs-muon-lr-scaling-laws`、`equiformer-v3-gradft-oom-rootcause`、`muon-lossweight-ablation-n2l2c64`
