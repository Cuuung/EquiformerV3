#!/usr/bin/env bash
###############################################################################
# GRAD-FT (stage-2) CONTINUATION from the AdamW-refined DIRECT checkpoint -- 16-GPU SenseCore job.
#
# LINEAGE:
#   ep20 peak (06-20 mlr@2e-4/35ep best_ckpt; fp32 cos 0.7664 / fmae 0.0253)
#     --AdamW low-lr DIRECT refine (15ep, lr1.5e-5)-->  07-02 adamw_refine best_ckpt  (this run's START)
#     --HybridMuon-moonlight GRAD-FT (10ep, mlr5e-5, fp32)-->  conservative-force model  (this run)
#
# WHY MUON IS OK HERE (unlike Scheme-S which diverged): grad-ft CHANGES the objective
#   (direct forces -> conservative -dE/dx autograd forces + virial stress), so the weights are
#   NOT at a converged minimum for the NEW loss -- there is a real gradient to descend, so Muon's
#   gradient-magnitude-blind constant step is a genuine re-optimization, not settling into an
#   existing basin. Muon has BEATEN AdamW on N2L2C64 grad-ft (fmae 0.0406->0.0343, -15.5%).
#
# RECIPE (identical to the proven stage2only gradft config): update_scale=moonlight, muon_lr=5e-5,
#   alr=5e-5, 10ep, loss e5-f10-s100, per-GPU bs=8 + grad_accum=4 (= global 512 on 16 GPU),
#   max_atoms=150, gradient_checkpointing=[1]x7. fp32 ONLY -- do NOT add --amp (amp crashes the
#   Wigner build in conservative-force mode: Float/Half index_put mismatch in so3.py).
#
# TIMING: fp32 double-backward is the heaviest stage; budget ~1.5-2x the direct rate -> 10ep ~ 1.5-2 days.
#
# CAVEAT ON THE START CKPT: the adamw_refine best_ckpt val was logged under --amp (fmae ~0.0264);
#   CKPT_A's 0.0253 was an fp32 re-eval, so they are NOT directly comparable. If an fp32 re-eval
#   shows the adamw_refine ckpt did NOT beat CKPT_A, swap START_CKPT below to CKPT_A:
#     /mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-20-03-16-16-muon_N7L4C128_direct_35ep_moonlight_mlr2e-4/best_checkpoint.pt
#
# LAUNCH: 16-GPU SenseCore job. Stop the OLD job first. Watch epoch 1 for growing SKIPPED / "Found nans"
#         (muon_lr is the knob). If OOM @step0 despite bs=8+max_atoms150, the next lever is a true
#         max_edges otf filter, NOT lower bs.
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
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[gradft-cont] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_CONT-from-adamw-refine_epochs@10-bs@8x16x4-maxatoms150_hybridmuon-moonlight-mlr@5e-5-alr@5e-5-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5_maxatoms150_bs8x16x4_from-adamw-refine'

# START = the 07-02 AdamW low-lr DIRECT refine best_checkpoint (weights-only, backbone+energy_block transfer).
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-02-01-36-00-adamw_refine_N7L4C128_direct_CONT_15ep_lr1.5e-5_from-ep20peak/best_checkpoint.pt'
##############################################################################

if [[ ! -s "$START_CKPT" ]]; then
  echo "[gradft-cont] ERROR: start checkpoint not found or empty: $START_CKPT" >&2
  exit 1
fi
echo "[gradft-cont] Finetuning (grad-ft) from: $START_CKPT"

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
# GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[gradft-cont] ===== MOONLIGHT GRAD-FT 10ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[gradft-cont] ===== DONE. Finetuned from: $START_CKPT ====="
