#!/usr/bin/env bash
###############################################################################
# GRAD-FT (stage-2) from the 35ep DIRECT run's best_checkpoint (= ep20 peak, "CKPT_A") -- 16-GPU job.
#
# This grad-ft starts DIRECTLY from the raw direct peak, SKIPPING the AdamW refine
# (that is a separate config/script). CKPT_A is a STRONGER direct start than the 25ep ckpt the
# first proven gradft used (fp32 val cos 0.7664 vs 0.7516).
#
# NO-OOM: recipe is byte-identical (bar the start ckpt) to the PROVEN gradft that ran 10ep clean
#   on 16 GPU (2026-06-23-04-09-36 run, wrote best_checkpoint.pt): bs=8/GPU + grad_accum=4 (global
#   512), max_atoms=150, gradient_checkpointing=[1]x7, fp32 (NO --amp). The start ckpt does not
#   change per-GPU memory, so it will not OOM where that one did not.
#
# fp32 ONLY -- do NOT add --amp (amp crashes the Wigner build in conservative-force mode:
#   Float/Half index_put mismatch in so3.py). Watch epoch 1 for growing SKIPPED / "Found nans"
#   (muon_lr is the knob). Stop the OLD job first.
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
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[gradft-A] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_CONT-from-35ep-direct-best_epochs@10-bs@8x16x4-maxatoms150_hybridmuon-moonlight-mlr@5e-5-alr@5e-5-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5_maxatoms150_bs8x16x4_from-35ep-direct-best'

# START = the 35ep DIRECT run's best_checkpoint (= ep20 peak / CKPT_A), weights-only.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-20-03-16-16-muon_N7L4C128_direct_35ep_moonlight_mlr2e-4/best_checkpoint.pt'
##############################################################################

if [[ ! -s "$START_CKPT" ]]; then
  echo "[gradft-A] ERROR: start checkpoint not found or empty: $START_CKPT" >&2
  exit 1
fi
echo "[gradft-A] Finetuning (grad-ft) from CKPT_A: $START_CKPT"

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
echo "[gradft-A] ===== MOONLIGHT GRAD-FT 10ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[gradft-A] ===== DONE. Finetuned from CKPT_A: $START_CKPT ====="
