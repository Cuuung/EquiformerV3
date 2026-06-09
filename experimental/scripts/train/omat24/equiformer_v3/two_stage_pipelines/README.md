# Two-stage chained training pipelines

Launcher scripts that run **Stage-1 (direct-force pretrain)** and, on success,
**Stage-2 (gradient-force finetune)** back-to-back inside a single SenseCore
GPU-pool job. The goal is **keeping cards busy** (no idle GPUs between stages):
one job reservation covers the whole direct -> grad-ft chain instead of two
separate submissions with a manual gap in between.

## How a pipeline script works
1. Stage-1 runs with a fixed `--identifier`; fairchem writes
   `$RUN_DIR/checkpoints/<timestamp>-<DIRECT_ID>/best_checkpoint.pt`.
2. `set -e` ensures Stage-1's `torchrun` must exit 0 before continuing.
3. The script globs the newest `*-<DIRECT_ID>/best_checkpoint.pt` and feeds it to
   Stage-2 via `--optim.load_pretrained_weights=<ckpt>` (overrides the yml path).
4. Stage-2 runs with its own `--identifier`.

Key invariants every script must preserve:
- **Stage-1 keeps `--amp`; Stage-2 must NOT use `--amp`** (gradient-force is
  fp32-only; amp crashes Wigner construction in so3.py).
- Both stages pass `--optim.num_workers=0` (lmdb env is not fork-safe).
- The whole script runs identically on every node; torchrun rendezvous via
  `$MASTER_ADDR/$MASTER_PORT` coordinates ranks, and the checkpoint lives on
  shared storage so all nodes see the same path.

## Naming convention
```
stage1-<stage1-desc>__stage2-<stage2-desc>.sh
```
- Double underscore `__` separates the two stages.
- Each `<...-desc>` is a compact summary: phase + arch/epoch + optimizer, e.g.
  `direct-N@2_L@2_C@64-15ep-hybridmuon` / `gradft-10ep-hybridmuon`.
- Keep the arch token (`N@2_L@2_C@64`) consistent with the yml filenames so the
  pipeline, the Stage-1 yml, and the Stage-2 yml are easy to cross-reference.

## Submit (SenseCore)
Point the job's launch command at the script (the `$SENSECORE_*` / `$MASTER_*`
env vars are injected by the pool and referenced inside):
```bash
bash experimental/scripts/train/omat24/equiformer_v3/two_stage_pipelines/<script>.sh
```

## Current scripts
- `stage1-direct-N@2_L@2_C@64-15ep-hybridmuon__stage2-gradft-10ep-hybridmuon.sh`
  - Stage-1: HybridMuon direct, N2L2C64, 15ep, 8-GPU (muon_lr 2e-2 / AdamW 2e-4).
  - Stage-2: HybridMuon grad-ft, 10ep, 8-GPU (muon_lr 5e-3 / AdamW 5e-5).
  - To make an AdamW variant, copy this script and swap `DIRECT_CFG/GRADFT_CFG`
    (and the `*_hybridmuon` tokens in the ids/filename) to the AdamW ymls.
