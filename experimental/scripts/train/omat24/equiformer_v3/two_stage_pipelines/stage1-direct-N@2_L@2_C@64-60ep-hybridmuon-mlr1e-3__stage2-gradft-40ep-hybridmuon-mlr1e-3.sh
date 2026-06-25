#!/usr/bin/env bash
###############################################################################
# Two-stage chained pipeline (intended for a 16-GPU SenseCore job):
#   Stage-1: HybridMuon direct-force pretrain, N2L2C64, 60ep, per-GPU 32
#            -> 16 GPU = global 512 (holds the 60ep recipe's lr pairing). muon_lr=1e-3.
#   Stage-2: HybridMuon gradient-force finetune, 40ep, per-GPU 16
#            -> 16 GPU = global 256. fp32, no --amp. muon_lr=1e-3.
#
# A/B Muon counterpart of the AdamW 60+40 (direct 60ep + gradft lr@5e-5-epochs@40) run.
#
# DIVERGENCE FIX (vs the mlr2e-3 pipeline): muon_lr 2e-3 -> 1e-3 ONLY. Everything else
# (warmup 0.5, clip_grad_norm 100, momentum 0.95, stress weight 100) is UNCHANGED.
#   Why: the mlr2e-3 run DIVERGED at epoch 36.1 / step 100224 (AMP GradScaler scale->0).
#   It was a METASTABLE mid-training blowup -- soft loss-spike @step53.6k (at lr=1.54e-3,
#   which is BELOW the 1.68e-3/2.0e-3 the run had already survived) -> high plateau ->
#   hard NaN crash @step100k. NOT an lr-schedule fault. 2e-3 looked safe only because the
#   earlier 15ep run never reached the accumulation regime. Halving muon_lr is the single
#   lever that shrinks every Muon update and pushes the spike-point back into the safe basin.
#   (clip is near-inert for the Muon group -- Newton-Schulz normalizes the update -- so it was
#   deliberately NOT changed; momentum/stress kept to preserve the A/B comparison.)
#   VERIFY: watch step 50k-55k for any spike recurrence; ideal = smooth loss, full 60 ep.
#   FALLBACK: if 1e-3 still spikes, override on the CLI without editing the yml, e.g.
#     append  --optim.optimizer_params.muon_lr=8e-4  to the Stage-1 torchrun below.
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

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
# ~/.bashrc's `proxy_on` exports http(s)_proxy=http://127.0.0.1:45889 and
# all_proxy=socks5h://127.0.0.1:45889 -- a LOOPBACK tunnel that only exists on the
# LOGIN machine. If `proxy_on` was active when you submitted, the job inherits it,
# but the compute node's 127.0.0.1:45889 is dead -> wandb's data-plane (file_stream
# uploads) fails with "unexpected EOF" / Client.Timeout, so the run shows on the web
# but every panel (incl. System) is empty. Compute nodes reach api.wandb.ai via their
# own native egress, so just drop the leaked proxy. (Plan B if a node has NO egress:
# WANDB_MODE=offline here + `wandb sync <run_dir>` from the login server afterward.)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the repo root has NO pyproject.toml, the container python has no pip,
# and the AdamW-era docker image bakes in an OLD fairchem (no HybridMuon branch) -- which is
# exactly why `optimizer: HybridMuon` crashed with:
#   AttributeError: module 'torch.optim' has no attribute 'HybridMuon'
# The cluster mounts this server into the job, so just point PYTHONPATH at this repo's src.
# PYTHONPATH is searched before site-packages, so the mounted new code overrides the image's
# baked-in old fairchem. (Muon needs NO new deps -- pure torch -- so no image rebuild.)
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
# Fail fast if the imported fairchem is STILL the wrong checkout (no HybridMuon branch).
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[pipeline] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

# --- Stage 1: HybridMuon direct-force pretrain (--amp ON) --------------------
# muon_lr=1e-3 (divergence fix vs the diverged 2e-3 run). global batch 32x16=512.
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@60-bs@32x16_hybridmuon-mlr@1e-3-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='muon_N2L2C64_direct_60ep_mlr1e-3'

# --- Stage 2: HybridMuon gradient-force finetune (fp32, NO --amp) ------------
# muon_lr=1e-3, 40ep, global batch 16x16=256.
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@2_L@2_C@64_attn-hidden@32_rbf@10_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_pt-dens-ft-no-reg_hybridmuon-mlr@1e-3-alr@5e-5-epochs@40-bs@16x16-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='muon_N2L2C64_gradft_40ep_mlr1e-3'
##############################################################################

# Honor SenseCore rendezvous when present (cluster GPU-pool job); else fall back to a
# single local node so the same script can be launched with a plain `bash <script>`.
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
