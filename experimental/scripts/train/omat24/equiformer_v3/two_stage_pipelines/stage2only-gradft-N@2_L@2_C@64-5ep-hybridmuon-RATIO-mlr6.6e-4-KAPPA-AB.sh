#!/usr/bin/env bash
###############################################################################
# STAGE-2 ONLY -- HybridMuon RATIO (Keller-Jordan) gradient-force finetune, 5ep.
#
# PURPOSE: build the MISSING ratio-side arm of a kappa_SRME A/B against
#   2026-06-29-16-57-36-abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100
# so we can finally ask whether update_scale (ratio vs moonlight) degrades the HIGH-ORDER
# derivative properties of the PES (thermal conductivity) while leaving E/F MAE a wash.
# kappa_SRME needs CONSERVATIVE forces (forces = -dE/dx), which only the grad-ft stage has --
# the direct-force stage-1 checkpoints are non-conservative and their kappa saturates near
# the V2 pathology (~1.6), so they CANNOT resolve this question.
#
# MODEL = N2L2C64 (2 layers, C=64, Lmax=2), the cheap proxy of the 30M recipe.
# Stage-2: conservative gradient force + stress, 5ep, per-GPU bs=16 + grad_accum=4
#          (= global 512 on 8 GPU). update_scale=RATIO, muon_lr=6.6e-4, alr=5e-5,
#          max_atoms=150. fp32, NO --amp.
#
# KNOWN CONFOUNDS (this is the CHEAP arm -- read before interpreting the result):
#   The stage-1 direct checkpoint below is ratio @ muon_lr=2e-3, NOT the calibration-matched
#   1e-3 (the established equivalence from the 60ep step-for-step A/B is ratio 1e-3 <->
#   moonlight 4e-4). So this arm differs from the moonlight arm in TWO ways, not one:
#     (1) update_scale ratio vs moonlight            <- the variable we want
#     (2) stage-1 ran at ~2x the matched step, and its direct val was actually BETTER
#         (fmae 0.0371 / cos 0.6596  vs  moonlight's 0.0392 / 0.6433)
#   => a kappa difference is SUGGESTIVE, not attributable. If it comes back interesting,
#      the clean follow-up is to re-run stage-1 as ratio @ mlr 1e-3 (15ep, ~7h/8GPU) and
#      repeat this stage-2 on top of it.
#   Everything else IS matched to the moonlight arm: bs 64x8 global 512 in stage-1,
#   bs 16x8x4 global 512 here, loss E:F:S = 5:10:100, max_atoms=150, guards, 5 epochs.
#
# >>> LAUNCH ON 8 GPU <<< (set device count = 8 in the job).
#     16x8x4 (bs x GPU x accum) = global 512. On a different GPU count the global batch --
#     and with it the muon_lr calibration -- shifts. Keep it at 8.
#   fp32-only: do NOT add --amp (amp crashes the Wigner build in conservative-force mode,
#   Float/Half index_put mismatch in so3.py). use_compile stays off (set in the yml).
# EXPECTED WALL CLOCK (8 GPU): ~3.5-7h (the moonlight partner's 5ep grad-ft reference).
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem (no HybridMuon).
# PYTHONPATH is searched before site-packages -> point it at this repo's src.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
# Reduce CUDA fragmentation (helps the memory-heavy fp32 conservative-force grad-ft stage).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[stage2] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@2_L@2_C@64_pt-dens-ft-no-reg_hybridmuon-ratio-mlr@6.6e-4-alr@5e-5-epochs@5-bs@16x8x4-maxatoms150-wd@1e-3_loss-e5-f10-s100_KAPPA-AB.yml'
GRADFT_ID='kappaAB_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_maxatoms150_bs16x8x4_loss-e5f10s100'

# Stage-1 (already DONE & healthy: ran 15/15 ep, val fmae 0.0371 / cos 0.6596) ckpt:
CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-13-04-01-04-muon_direct_15ep_8card_mlr2e-3/best_checkpoint.pt'
##############################################################################

if [[ ! -s "$CKPT" ]]; then
  echo "[stage2] ERROR: stage-1 checkpoint not found or empty: $CKPT" >&2
  exit 1
fi
echo "[stage2] Finetuning from stage-1 ckpt: $CKPT"

TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
MAIN_COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1
  --optim.num_workers=0
)

###############################################################################
# Stage 2: RATIO GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[stage2] ===== RATIO GRAD-FT finetune 5ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[stage2] ===== DONE. Finetuned from: $CKPT ====="
