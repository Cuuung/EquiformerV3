#!/usr/bin/env bash
###############################################################################
# VALIDATE-ONLY (no training): measure the REAL val metrics of two 30M direct
# checkpoints under one identical eval path, so mae/cos are directly comparable.
#
#   CKPT_A = 06-20 mlr@2e-4 35ep run's best_checkpoint.pt
#            (ep~20 pre-collapse peak; memory says val cos ~0.756. best_checkpoint
#             is selected by primary_metric=forces_mae, so it is ALSO that run's
#             min-forces_mae epoch. THIS run wants its exact mae/cos as a TARGET
#             number -- we are NOT adopting the ckpt itself, see note below.)
#   CKPT_B = 06-22 mlr@1.5e-4 25ep run's best_checkpoint.pt
#            (first healthy full 30M run; recorded val cos 0.7516 / fmae 0.0273).
#
# WHY validate-only and not just trust the logged numbers: recompute both under
# the SAME code / same fp32 eval / same val split so the A-vs-B delta is apples-to-
# apples (the two runs logged at different times, different amp state).
#
# WHY we are NOT using CKPT_A for the pipeline (user's call): it is a pre-collapse
# snapshot of a run that then diverged -- not reproducible / not engineering-coherent.
# We only want its metric as the bar a clean, reproducible run should clear.
#
# Model arch + val set are IDENTICAL across all N7L4C128 direct configs, so ONE
# config validates both ckpts. We reuse the proven 25ep direct yml (its `val:` block
# = /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/val, present on this share).
#
# RUN: 1 GPU is plenty (val set is small). Launch on a GPU node with the training
#      image (this repo's src on PYTHONPATH), or submit as a 1-GPU SenseCore job.
#      fp32 (NO --amp): AMP-off inference is the standard eval path here.
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# Strip the login-node loopback proxy (dead on a compute node) + disable wandb for a
# throwaway eval so it can't hang on network egress.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export WANDB_MODE=disabled

# Use THIS repo's fairchem (has HybridMuon), not the image's old baked-in copy.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported: '+bt.__file__; print('[val] fairchem OK:', bt.__file__)"

CKDIR=/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints
CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@25-bs@32x16_hybridmuon-moonlight-mlr@1.5e-4-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml'
VAL_RUN_DIR=/mnt/afs/share/checkpoint/equiformerV3/yaolekai/_eval

CKPT_A="$CKDIR/2026-06-20-03-16-16-muon_N7L4C128_direct_35ep_moonlight_mlr2e-4/best_checkpoint.pt"
CKPT_B="$CKDIR/2026-06-22-03-29-04-muon_N7L4C128_direct_25ep_moonlight_mlr1.5e-4/best_checkpoint.pt"

run_val () {
  local tag="$1" ckpt="$2"
  echo "=================================================================="
  echo "[val] $tag  <-  $ckpt"
  echo "=================================================================="
  [[ -s "$ckpt" ]] || { echo "[val] MISSING/empty: $ckpt" >&2; return 1; }
  torchrun --nproc_per_node 1 --master_port "${MASTER_PORT:-29533}" \
    my_main.py --mode validate --config-yml "$CFG" \
    --checkpoint "$ckpt" \
    --num-gpus 1 --num-nodes 1 \
    --run-dir "$VAL_RUN_DIR" \
    --print-every 50 --seed 1 \
    --optim.num_workers=0 \
    --identifier "val_${tag}"
  # NOTE: if restoring the HybridMuon optimizer state from --checkpoint errors on a
  # pure eval, swap the two lines above for a weights-only load:
  #     my_main.py --mode validate --config-yml "$CFG" \
  #     --optim.load_pretrained_weights="$ckpt" \
  # (load_pretrained_weights loads MODEL weights only, no optimizer/scheduler state.)
}

run_val "A_2e-4_ep20peak"   "$CKPT_A"
run_val "B_1.5e-4_25ep"     "$CKPT_B"

echo "[val] DONE. Compare the two blocks' forces_mae / forces_cosine_similarity / energy_mae above."
