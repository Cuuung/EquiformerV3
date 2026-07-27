#!/usr/bin/env bash
###############################################################################
# PLACEHOLDER / GPU-HOLD job (2 node x 4 GPU = 8 ranks).
#
# PURPOSE: purely occupy the allocation by running a real training loop. It produces
#   NOTHING of value on purpose:
#     - wandb OFF (WANDB_MODE=disabled)          -> no experiment logging
#     - run-dir -> a throwaway _placeholder dir  -> shared checkpoint tree stays clean
#     - eval / checkpoint interval set huge       -> effectively no disk output
#     - print-every huge                          -> almost no stdout
#     - max_epochs huge                           -> runs until you kill it (or job timeout)
#
# It reuses the E-G config (fp32-eager gradft from the A0 direct bf16 ckpt) only because that
# path is already validated to train stably. This is NOT the E-G experiment -- that one already
# ran and its kappa is in docs/BF16_DIRECT_KAPPA_REGRESSION.md; its script is left untouched.
#
# LAUNCH (one line):  bash "<abs path to this file>"
# STOP: kill the SenseCore job when you no longer need the hold.
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"
# No experiment logging.
export WANDB_MODE=disabled
export WANDB_SILENT=true
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[placeholder] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/dpa4_ablation/dpa4_A0-baseline_N@2_L@2_C@64_gradft-5ep_moonshot_GRADFP32-eager.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/_placeholder'   # throwaway; safe to delete
ID='PLACEHOLDER_hold_2node4gpu'
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-23-05-30-40-dpa4_A0-baseline_N2L2C64_direct_15ep_moonshot_bf16compile/best_checkpoint.pt'

if [[ ! -s "$START_CKPT" ]]; then
  echo "[placeholder] ERROR: start ckpt not found: $START_CKPT" >&2; exit 1
fi
mkdir -p "$RUN_DIR"
echo "[placeholder] ===== holding 2node x 4GPU; no wandb, no meaningful output. Kill the job to release. ====="

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
  --print-every 100000 --seed 1 --optim.num_workers=0 \
  --optim.max_epochs=100000 \
  --optim.eval_every=100000000 \
  --optim.checkpoint_every=100000000 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[placeholder] ===== exited ====="
