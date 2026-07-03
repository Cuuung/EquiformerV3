#!/usr/bin/env bash
###############################################################################
# SCHEME S -- low-lr DIRECT CONTINUATION from the ep20 peak (16-GPU SenseCore job).
#
#   GOAL: refine an already-converged 30M direct checkpoint at a low muon_lr to squeeze
#   the last few % of force-mae and probe the MP CPS gap to SOTA (~1.7% CPS behind ->
#   a ~3% fmae gain is large, not marginal). This is an INTERNAL probe, NOT a publishable
#   pipeline (a paper needs an architecture-level early-stop, not a diverged run's frozen best).
#
#   START = the 06-20 mlr@2e-4/35ep run's best_checkpoint.pt = its ep20 PRE-COLLAPSE PEAK
#           (fp32 re-eval: forces_cos 0.7664 / forces_mae 0.0253). Loaded WEIGHTS-ONLY via
#           --optim.load_pretrained_weights -> Muon momentum + EMA re-init from those weights,
#           so this run does NOT inherit the momentum that fed the original ep21 runaway.
#
#   WHY SAFE: in the original 35ep cosine the EFFECTIVE muon_lr at the ep20 peak was
#   2e-4 * 0.5*(1+cos(pi*20/35)) ~= 7.8e-5, and the ep21 runaway happened AT that lr WITH
#   20ep of accumulated momentum. This continuation starts at muon_lr=5e-5 (BELOW 7.8e-5),
#   fresh (zero) momentum, cosine decay to ~0 over 18ep -> even the hottest point (the start)
#   is cooler than the lr at which the original diverged. Danger band = the FIRST ~3-4 epochs
#   (hottest), NOT the tail. WATCH max NS-input RMS ep0-4: if it creeps up, detune to 3e-5.
#
#   alr (AdamW non-matrix lr) is dropped 2e-4 -> 5e-5 IN LOCK-STEP with muon_lr (else the
#   AdamW branch keeps shoving bias/norm while Muon barely moves -> asymmetric drift out of basin).
#
# TIMING: ~0.94 h/epoch on 16 GPU (32x16 = global 512, --amp) -> 18ep ~= 17h (fits the 14-18h target).
#
# LAUNCH: 16-GPU SenseCore job. Stop the OLD job first (fixed muon.py + this lr read at startup
#         via PYTHONPATH src). --amp ON (direct force = fp16, same regime as the peak).
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# Strip the login-node loopback proxy (dead on a compute node) so wandb egress works.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the image bakes an OLD fairchem (no HybridMuon). PYTHONPATH is searched
# before site-packages -> point it at this repo's src to override the image.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon): '+bt.__file__; print('[schemeS] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_CONT-from-ep20peak_epochs@18-bs@32x16_hybridmuon-moonlight-mlr@5e-5-alr@5e-5-wd@1e-3-warmup@0.3_dens-no-stress_loss-e5-f10-s100.yml'
ID='muon_N7L4C128_direct_CONT_schemeS_18ep_moonlight_mlr5e-5_from-ep20peak'

# The ep20 peak to continue FROM (weights-only). This is CKPT_A from the A-vs-B fp32 re-eval.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-20-03-16-16-muon_N7L4C128_direct_35ep_moonlight_mlr2e-4/best_checkpoint.pt'
##############################################################################

[[ -s "$START_CKPT" ]] || { echo "[schemeS] ERROR: start checkpoint missing/empty: $START_CKPT" >&2; exit 1; }
echo "[schemeS] continuing FROM (weights-only): $START_CKPT"

# Honor SenseCore rendezvous when present; else fall back to a single local node.
TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)

echo "[schemeS] ===== low-lr DIRECT continuation 18ep ($ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$CFG" --amp \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --run-dir   "$RUN_DIR" \
  --print-every 200 --seed 1 \
  --optim.num_workers=0 \
  --identifier "$ID" \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[schemeS] ===== DONE. best_checkpoint under $RUN_DIR/checkpoints/*-$ID/ ====="
