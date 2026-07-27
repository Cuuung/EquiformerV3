#!/usr/bin/env bash
###############################################################################
# KAPPA-AB (clean single-variable): keller gradft on the SAME 15ep moonshot direct base
#   as ablation-A (moonshot gradft), so keller vs moonshot differ ONLY in the gradft stage.
#
# WHY: the existing keller N2L2C64 (2026-07-20 ...RATIO_mlr6.6e-4) sat on a DIFFERENT direct base
#   (2026-06-13 muon_direct_15ep mlr2e-3, update_scale=default), while ablation-A moonshot sat on
#   2026-06-29 abl_...moonlight. Different direct bases -> the kappa gap could not be attributed to
#   the optimizer. This run fixes that: SAME direct base, only the gradft update_scale + muon_lr differ.
#
# CLEAN A/B -- config differs from the ablation-A gradft config in EXACTLY 3 lines:
#     update_scale:            moonlight -> ratio       (algorithm knob #1)
#     muon_lr:                 1.5e-4    -> 6.6e-4       (MATCHED effective step, not cranked:
#                                keller RMS = lr/sqrt(fan_in) vs moonshot 0.2*lr; docs say 6.6e-4 ~= 1.5e-4)
#     load_pretrained_weights: (stale 60ep) -> the 15ep moonshot direct base below
#   Everything else byte-identical: 5ep, loss e5-f10-s100, per-GPU bs 16 + grad_accum 4 (= global 512),
#   max_atoms 150, weight_decay 1e-3, alr(lr_initial) 5e-5, fp32 (no --amp, no compile), direct_prediction False.
#
# TOPOLOGY: 4 node x 2 GPU = 8 ranks. per-GPU bs 16 x 8 ranks x accum 4 = GLOBAL 512 (UNCHANGED from
#   the original 8-card layout -- only spread 8 GPUs across 4 nodes; the training math is identical,
#   just extra inter-node comm). `--num-gpus` is PER NODE (fairchem gpus_per_node), hence 2 / 4.
#
# fp32 ONLY -- do NOT add --amp (amp crashes the Wigner build in conservative-force mode).
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[keller-kappaAB] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/dpa4_ablation/keller_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_from-abl-moonlight15ep-direct_KAPPA-AB.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='keller_N2L2C64_gradft_5ep_RATIO_mlr6.6e-4_from-abl-moonlight15ep-direct_KAPPA-AB'

# SAME 15ep moonshot direct base as ablation-A. Pinned in the yml too; passed here for an explicit
# launch-log record and to override any stale value unambiguously.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-29-09-38-08-abl_N2L2C64_direct_15ep_moonlight_mlr4e-4_loss-e5f10s100/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[keller-kappaAB] ERROR: 15ep moonshot direct base not found: $START_CKPT" >&2; exit 1
fi
echo "[keller-kappaAB] ===== keller gradft 5ep ($ID) from 15ep moonshot direct base ====="
echo "[keller-kappaAB] direct base: $START_CKPT"

torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-2}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-4}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-2}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-4}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[keller-kappaAB] ===== DONE. Send best_checkpoint.pt to phonon/relax -> kappa_SRME."
echo "[keller-kappaAB]       Compare vs ablation-A moonshot gradft (same 15ep direct base): kappa 0.4638 / CPS 0.7302."
