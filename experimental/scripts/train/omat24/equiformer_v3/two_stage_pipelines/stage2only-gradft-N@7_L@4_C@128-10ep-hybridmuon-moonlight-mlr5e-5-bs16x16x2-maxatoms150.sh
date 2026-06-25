#!/usr/bin/env bash
###############################################################################
# STAGE-2 ONLY -- HybridMuon MOONLIGHT gradient-force finetune (10ep), resuming
# from the ALREADY-SUCCESSFUL stage-1 direct checkpoint (skip stage-1; do NOT
# re-run the 25ep direct, it completed healthy: val cos 0.7516 @ ep25).
#
# MODEL = paper's ~30M MPtrj model (Table 10): Lmax=4, 7 layers, C=128 (N7L4C128).
# Stage-2: conservative gradient force (forces=-dE/dx, double backward) + stress,
#          10ep, per-GPU bs=16 + grad_accum=2 (= global 512 on 16 GPU).
#          update_scale=moonlight, muon_lr=5e-5. fp32, NO --amp.
#
# WHY THIS RUN (the OOM fix):
#   Prior 06-23 gradft OOMed at STEP 0 in the conservative double-backward even at
#   bs=16. ROOT CAUSE was NOT bs -- it was an INERT structure-size guard: the yml had
#   max_atoms=24000, but MPtrj's largest structure is only 444 atoms (mean 31, p99 144),
#   so 24000 filtered ZERO structures and the 150-444 atom monsters blew the fp32
#   double-backward peak. The paper excludes >24000 EDGES; we proxy in atom-space.
#   FIX (config-level, in the yml): max_atoms 24000 -> 150 (drops only 0.82% = 11.7k/1.42M,
#   the memory tail), keeping bs=16 + grad_accum=2 + gradient_checkpointing=[1]x7.
#   UPDATE: max_atoms=150 alone STILL OOMd @step0 at bs=16 (recompute, 259MB free) -> the per-sample
#   cap could not close the ~2x gap to the paper's bs=32. NOW bs 16->8 + grad_accum 2->4 (= global 512),
#   which deterministically halves peak activation. max_atoms=150 kept. If bs=8 STILL OOMs, the next
#   lever is a TRUE max_edges otf filter (none exists yet), not lower bs.
#
# >>> LAUNCH ON 16 GPU <<< (set device count = 16 in the job).
#     16x16x2 (bs x GPU x accum) = global 512. On a different GPU count global batch shifts.
#   NOTE: launch AFTER the old job is stopped. fixed muon.py + the new max_atoms read at
#   startup via PYTHONPATH=src. fp32-only: do NOT add --amp (amp crashes the Wigner build
#   in conservative-force mode -- Float/Half index_put mismatch in so3.py).
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

GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_attn-hidden@32_rbf@10_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_pt-dens-ft-no-reg_hybridmuon-moonlight-mlr@5e-5-alr@5e-5-epochs@10-bs@16x16x2-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5_maxatoms150_bs8x16x4'

# Stage-1 (already DONE & healthy) best checkpoint to finetune from:
CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-22-03-29-04-muon_N7L4C128_direct_25ep_moonlight_mlr1.5e-4/best_checkpoint.pt'
##############################################################################

if [[ ! -s "$CKPT" ]]; then
  echo "[stage2] ERROR: stage-1 checkpoint not found or empty: $CKPT" >&2
  exit 1
fi
echo "[stage2] Finetuning from stage-1 ckpt: $CKPT"

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
# Stage 2: GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[stage2] ===== MOONLIGHT GRAD-FT finetune 10ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[stage2] ===== DONE. Finetuned from: $CKPT ====="
