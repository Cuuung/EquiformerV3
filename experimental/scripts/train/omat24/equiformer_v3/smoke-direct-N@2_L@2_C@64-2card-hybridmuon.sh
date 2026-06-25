#!/usr/bin/env bash
###############################################################################
# 2-GPU SMOKE TEST: HybridMuon direct-force pretrain, N2L2C64.
#
# Purpose: watch the loss curve of the stabilized lever set BEFORE committing a
# full multi-card run. Levers vs the diverged mlr@1e-2 config:
#   - muon_lr   0.01 -> 0.005   (降低lr)
#   - warmup    epochs 0.5, factor 1e-3   (采用warmup)
#   - grad clip 100  -> 1.0     (开启grad clip; 100 was effectively off)
#   - batch     64   -> 32      (减小batch)
# The 8-GPU mlr@1e-2 run diverged at epoch ~1.7, so this config runs 2 epochs to
# reach that region. Tune batch_size / muon_lr from the curve, then scale to the
# 8-card (15ep) and 16-card (60ep) pipelines.
#
# Runs as a 2-proc single-node torchrun. If SENSECORE_* env vars are present
# (cluster job) they are honored; otherwise it falls back to localhost:2.
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
# ~/.bashrc's `proxy_on` exports http(s)_proxy=http://127.0.0.1:45889 and
# all_proxy=socks5h://127.0.0.1:45889 -- a LOOPBACK tunnel that only exists on the
# LOGIN machine. On a compute node 127.0.0.1:45889 is dead -> wandb's data-plane
# dies ("unexpected EOF"/Client.Timeout) so the run shows but every panel is empty.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem (no HybridMuon) ->
# `optimizer: HybridMuon` would crash with AttributeError. PYTHONPATH is searched
# before site-packages, so point it at this repo's src to override the image.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[smoke] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
NGPU=2

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@2-bs@32x2_hybridmuon-mlr@5e-3-alr@2e-4-wd@1e-3-warmup@0.5-clip@1_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='muon_smoke_2card_mlr5e-3_clip1'
##############################################################################

# Honor SenseCore rendezvous if present (cluster job); else single-node localhost.
TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-$NGPU}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
MAIN_COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-$NGPU}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1
  --optim.num_workers=0
)

echo "[smoke] ===== 2-card DIRECT smoke ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[smoke] ===== DONE ====="
