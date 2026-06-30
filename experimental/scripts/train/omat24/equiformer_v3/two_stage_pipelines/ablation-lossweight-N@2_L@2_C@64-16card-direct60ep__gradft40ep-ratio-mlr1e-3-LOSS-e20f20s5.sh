#!/usr/bin/env bash
###############################################################################
# LOSS-WEIGHT ABLATION (arm B: E:F:S = 20:20:5, paper config) -- 16-GPU job.
#
# A/B vs the existing 5:10:100 baseline (the 06-16 runs:
#   direct  checkpoints/2026-06-16-02-54-56-muon_N2L2C64_direct_60ep_mlr1e-3
#   gradft  checkpoints/2026-06-16-18-57-04-muon_N2L2C64_gradft_40ep_mlr1e-3 ).
# This is a CLEAN single-variable ablation: this pipeline is byte-identical to
#   stage1-direct-...-60ep-...-mlr1e-3__stage2-gradft-40ep-...-mlr1e-3.sh
# EXCEPT the loss weights (5:10:100 -> 20:20:5) and the run identifiers. Same
# optimizer (HybridMuon ratio, muon_lr=1e-3), same 60+40 epochs, same 16-GPU
# batch layout (direct 32x16=global 512, gradft 16x16=global 256), same warmup/
# clip/momentum. So you do NOT need to re-run the baseline -- reuse the 06-16 run.
#
# Why this is safe at the same muon_lr: Muon orthogonalizes the update (Newton-
# Schulz), so the step MAGNITUDE is set by lr*scale and is ~decoupled from the
# gradient magnitude. Re-weighting the loss (5:10:100 -> 20:20:5) changes the
# E/F/S DIRECTION balance, not the Muon step size -> mlr@1e-3 stays valid.
#
# NOTE on the divergence guard: the new yml sets spike_abs_threshold: null. The
# 06-16 baseline ran on the PRE-GUARD muon.py (no abs backstop); the current code
# DEFAULTS that knob to 4.0 (a 30M/moonlight tuning) which would false-abort this
# ratio/mlr@1e-3 run. null restores the baseline's behavior. skip_nonfinite +
# 8x-EMA relative guard stay on. If 1e-3 ever spikes, override on the CLI:
#   append  --optim.optimizer_params.muon_lr=8e-4  to the Stage-1 torchrun below.
#
# Chaining (identical to the baseline pipeline):
#   1. Stage-1 (direct, --amp) writes to $RUN_DIR/checkpoints/<ts>-$DIRECT_ID/best_checkpoint.pt
#   2. We glob the newest match on $DIRECT_ID and feed best_checkpoint.pt to
#      Stage-2 via --optim.load_pretrained_weights=<ckpt>.
#   3. Stage-1 keeps --amp; Stage-2 must NOT use --amp (conservative force is fp32-only).
# GPU count is taken from $SENSECORE_ACCELERATE_DEVICE_COUNT (set 16 in the job).
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: point PYTHONPATH at this repo's src so the mounted new code
# (HybridMuon branch) overrides the image's baked-in old fairchem. Muon needs no new deps.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[pipeline] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

# --- Stage 1: HybridMuon direct-force pretrain (--amp ON) --------------------
# ratio muon_lr=1e-3, 60ep, global batch 32x16=512. LOSS 20:20:5.
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@60-bs@32x16_hybridmuon-mlr@1e-3-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e20-f20-s5_ablation.yml'
DIRECT_ID='abl_N2L2C64_direct_60ep_mlr1e-3_loss-e20f20s5'

# --- Stage 2: HybridMuon gradient-force finetune (fp32, NO --amp) ------------
# ratio muon_lr=1e-3, 40ep, global batch 16x16=256. LOSS 20:20:5.
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@2_L@2_C@64_attn-hidden@32_rbf@10_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_pt-dens-ft-no-reg_hybridmuon-mlr@1e-3-alr@5e-5-epochs@40-bs@16x16-wd@1e-3_loss-e20-f20-s5_ablation.yml'
GRADFT_ID='abl_N2L2C64_gradft_40ep_mlr1e-3_loss-e20f20s5'
##############################################################################

# Honor SenseCore rendezvous when present; else fall back to a single local node.
TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
MAIN_COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}"
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
