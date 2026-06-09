#!/usr/bin/env bash
###############################################################################
# Two-stage chained pipeline (intended for a 16-GPU SenseCore job):
#   Stage-1: HybridMuon direct-force pretrain, N2L2C64, 60ep, per-GPU 32
#            -> 16 GPU = global 512 (holds the 60ep recipe's lr pairing).
#   Stage-2: HybridMuon gradient-force finetune, 25ep, per-GPU 16
#            -> 16 GPU = global 256. fp32, no --amp.
#
# Chaining:
#   1. Stage-1 runs with a fixed --identifier ($DIRECT_ID). fairchem writes its
#      checkpoint to $RUN_DIR/checkpoints/<timestamp>-$DIRECT_ID/best_checkpoint.pt
#   2. After Stage-1's torchrun returns 0, we glob the newest match on $DIRECT_ID
#      and feed best_checkpoint.pt to Stage-2 via the CLI override
#      --optim.load_pretrained_weights=<ckpt> (overrides the yml's placeholder).
#   3. Stage-1 keeps --amp; Stage-2 must NOT use --amp (amp crashes Wigner build
#      in conservative-force mode -- gradient force is fp32-only).
#
# Runs identically on every node; torchrun rendezvous via $MASTER_ADDR/$MASTER_PORT
# coordinates ranks, and the checkpoint lives on shared storage (all nodes see it).
# GPU count is taken from $SENSECORE_ACCELERATE_DEVICE_COUNT (set 16 in the job).
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3
# The repo root has NO pyproject.toml/setup.py; the installable package is
# packages/fairchem-core (its src/ is a symlink to ../../src). Install by that
# explicit path so it works regardless of cwd (plain `pip install -e .` from the
# repo root fails: "does not appear to be a Python project").
pip install -e packages/fairchem-core --no-deps

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

# --- Stage 1: HybridMuon direct-force pretrain (--amp ON) --------------------
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@60-bs@32x16_hybridmuon-mlr@2e-2-alr@2e-4-wd@1e-3_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='mptrj_direct_N@2_L@2_C@64_60ep_hybridmuon'

# --- Stage 2: HybridMuon gradient-force finetune (fp32, NO --amp) ------------
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@2_L@2_C@64_attn-hidden@32_rbf@10_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_pt-dens-ft-no-reg_hybridmuon-mlr@5e-3-alr@5e-5-epochs@25-bs@16x16-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='mptrj_grad-ft_N@2_L@2_C@64_25ep_hybridmuon_from-60ep-hybridmuon'
##############################################################################

TORCHRUN_COMMON=(
  --nproc_per_node "$SENSECORE_ACCELERATE_DEVICE_COUNT"
  --nnodes        "$SENSECORE_PYTORCH_NNODES"
  --node_rank     "$SENSECORE_PYTORCH_NODE_RANK"
  --master_addr   "$MASTER_ADDR"
  --master_port   "$MASTER_PORT"
)
MAIN_COMMON=(
  --num-gpus  "$SENSECORE_ACCELERATE_DEVICE_COUNT"
  --num-nodes "$SENSECORE_PYTORCH_NNODES"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1
  --optim.num_workers=0
)

###############################################################################
# Stage 1: DIRECT  (--amp ON)
###############################################################################
echo "[two-stage] ===== Stage 1: DIRECT pretrain ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

###############################################################################
# Locate the checkpoint Stage 1 just produced
###############################################################################
echo "[two-stage] Stage 1 finished. Locating its best_checkpoint.pt ..."
CKPT=$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT:-}" || ! -s "$CKPT" ]]; then
  echo "[two-stage] ERROR: could not find a non-empty best_checkpoint.pt for identifier '$DIRECT_ID'." >&2
  echo "[two-stage] Looked under: $RUN_DIR/checkpoints/*-$DIRECT_ID/best_checkpoint.pt" >&2
  exit 1
fi
echo "[two-stage] Using pretrained weights: $CKPT"

###############################################################################
# Stage 2: GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[two-stage] ===== Stage 2: GRAD-FT finetune ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[two-stage] ===== DONE. Stage-1 ckpt: $CKPT ====="
