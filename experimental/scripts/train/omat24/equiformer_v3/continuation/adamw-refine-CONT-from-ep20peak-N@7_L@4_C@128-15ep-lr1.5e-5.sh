#!/usr/bin/env bash
###############################################################################
# ADAMW LOW-LR REFINEMENT ("polish") -- DIRECT continuation from the ep20 peak (16-GPU SenseCore job).
#
#   WHY THIS EXISTS: the prior Scheme-S run tried to refine this SAME ep20 peak with HybridMuon
#   @ muon_lr=5e-5 (below the ~7.8e-5 at which the original run diverged, fresh momentum) and STILL
#   blew up -- it died at ep~2.7 (first val already NaN, no best_checkpoint written) and hard-aborted
#   at ep5.51/step15300 (AMP GradScaler->0). ROOT CAUSE: Muon's update is gradient-magnitude-INDEPENDENT,
#   so on an already-converged sharp minimum it takes constant-size steps and walks the weights OUT of
#   the basin -> fp16 overflow -> NaN. Lowering muon_lr only delays it. => Muon cannot refine a converged
#   minimum. AdamW's per-coordinate step SELF-ANNEALS (step -> 0 as gradients -> 0), so it can settle
#   into the basin instead of wandering out. This run switches the optimizer to AdamW at a low lr.
#   (See memory: muon-divergence-lessons.)
#
#   START = the 06-20 mlr@2e-4/35ep run's best_checkpoint.pt = its ep20 PRE-COLLAPSE PEAK
#           (fp32 re-eval: forces_cos 0.7664 / forces_mae 0.0253). Loaded WEIGHTS-ONLY via
#           --optim.load_pretrained_weights -> no optimizer/scheduler state inherited.
#           (This checkpoint is UNTOUCHED by the failed Scheme-S run.)
#
#   RECIPE: AdamW lr=1.5e-5 (cosine to ~1.5e-8 over 15ep), warmup 0.3ep, DeNS OFF, loss e5-f10-s100,
#           global batch 512 (32x16). GOAL: squeeze the last few % of force-mae to probe the MP CPS
#           gap to SOTA (we are only ~1.7% CPS behind). INTERNAL probe, NOT a publishable pipeline.
#
# TIMING: ~0.85-0.94 h/epoch on 16 GPU (DeNS off is slightly faster) -> 15ep ~= 13-14h.
#
# LAUNCH: 16-GPU SenseCore job. Stop the OLD job first. --amp ON (direct force = fp16, same as the peak);
#         AdamW staying in-basin means the fp16 overflow that killed Scheme-S should not recur. If any
#         "Found nans while computing loss" appears, fall back to fp32 by removing the --amp flag below.
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# Strip the login-node loopback proxy (dead on a compute node) so wandb egress works.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the image bakes an OLD fairchem (no HybridMuon). PYTHONPATH is searched
# before site-packages -> point it at this repo's src to override the image.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import fairchem.core.trainers.base_trainer as bt; print('[adamw-refine] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_CONT-from-ep20peak_epochs@15-bs@32x16_adamw-lr@1.5e-5-wd@1e-3-warmup@0.3_no-dens_loss-e5-f10-s100.yml'
ID='adamw_refine_N7L4C128_direct_CONT_15ep_lr1.5e-5_from-ep20peak'

# The ep20 peak to continue FROM (weights-only). CKPT_A from the A-vs-B fp32 re-eval; untouched by Scheme-S.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-20-03-16-16-muon_N7L4C128_direct_35ep_moonlight_mlr2e-4/best_checkpoint.pt'
##############################################################################

[[ -s "$START_CKPT" ]] || { echo "[adamw-refine] ERROR: start checkpoint missing/empty: $START_CKPT" >&2; exit 1; }
echo "[adamw-refine] continuing FROM (weights-only): $START_CKPT"

# Honor SenseCore rendezvous when present; else fall back to a single local node.
TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)

echo "[adamw-refine] ===== AdamW low-lr DIRECT refinement 15ep ($ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$CFG" --amp \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --run-dir   "$RUN_DIR" \
  --print-every 200 --seed 1 \
  --optim.num_workers=0 \
  --identifier "$ID" \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[adamw-refine] ===== DONE. best_checkpoint under $RUN_DIR/checkpoints/*-$ID/ ====="
