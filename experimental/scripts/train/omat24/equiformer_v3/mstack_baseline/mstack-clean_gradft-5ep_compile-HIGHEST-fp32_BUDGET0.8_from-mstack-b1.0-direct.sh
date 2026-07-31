#!/usr/bin/env bash
###############################################################################
# GRAD BUDGET ablation (grad-only): highest(fp32) + compile, budget 0.6 -> 0.8.
#
# The already-finished mstack-clean grad = compile + HIGHEST(fp32) + budget 0.6 (kappa 0.4471).
#   This run is IDENTICAL to it except EQV3_ACT_MEM_BUDGET 0.6 -> 0.8. Same HIGHEST config, same
#   compile ON, same mstack b1.0 direct base, same optimizer, same 5ep. => a clean single-variable
#   test of the grad-stage activation-memory budget at fp32.
#
# WHY: budget was proven NOT math-neutral for bf16 DIRECT (0.6 cost -16.6% forces_mae + big kappa).
#   Grad is fp32 (23-bit mantissa) so recompute should be near-bit-faithful and budget ~neutral --
#   but after the direct surprise we verify it directly on the grad stage. 0.6 -> 0.8 = fewer
#   activations recomputed. Compare final MAE + kappa vs the budget-0.6 highest run (0.4471).
#     match     -> grad budget is math-neutral (as expected for fp32); safe to tune for memory.
#     0.8 better -> grad recompute also perturbs fp32 slightly; prefer higher budget when it fits.
#
# GRAD-ONLY: loads the frozen mstack b1.0 direct best_checkpoint (pinned in the yml AND passed on
#   the CLI). CONFIG is REUSED byte-identically from the HIGHEST run -- budget is an env var, not a
#   yml field, so the only difference between this run and the 0.6 run is the export value below.
#
# TOPOLOGY: 2 node x 4 GPU = 8 ranks. `--num-gpus` is PER NODE (fairchem gpus_per_node), hence 4/2.
#   grad bs 16 x 8 ranks x accum 4 = GLOBAL 512 (same as the 0.6 run -- comparison stays valid).
# NO --amp (grad amp crashes Wigner); grad blocks are fp32.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[grad-budget0.8] fairchem OK:', bt.__file__)"

# REUSED byte-identically from the HIGHEST (budget 0.6) run -- budget is env-only, not in the yml.
CFG='experimental/configs/omat24/mptrj/experiments/gradient/mstack_baseline/mstack-clean_N@2_L@2_C@64_gradft-5ep_moonshot_compile-HIGHEST-fp32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32_BUDGET0.8'

# SAME b1.0 direct base as the budget-0.6 highest run -- mandatory for the isolation. Pinned in yml too.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-27-12-41-36-mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[grad-budget0.8] ERROR: mstack b1.0 direct base not found: $START_CKPT" >&2; exit 1
fi
echo "[grad-budget0.8] ===== highest + budget 0.8 ($ID) from mstack b1.0 direct base ====="
echo "[grad-budget0.8] direct base: $START_CKPT"

EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.8}" \
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-2}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-2}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[grad-budget0.8] ===== DONE. Compare vs the budget-0.6 highest run (mstack grad):"
echo "[grad-budget0.8]   MAE:   forces_mae 0.04268 / cos 0.6419 ;   kappa: 0.4471."
echo "[grad-budget0.8]   match => grad budget math-neutral (fp32);  0.8 better => grad recompute perturbs fp32 too."
