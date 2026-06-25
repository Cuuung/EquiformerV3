#!/usr/bin/env bash
###############################################################################
# 8-GPU HybridMuon direct-force pretrain, N2L2C64, 15ep.
#
# A/B Muon counterpart of the AdamW 15ep direct config:
#   ...epochs@15-bs@64x8-lr@2e-4-wd@1e-3_dens-no-stress_loss-e5-f10-s100.yml
# EVERYTHING is identical to that AdamW config (model / data / loss e5-f10-s100 /
# 15ep / per-GPU batch 64 -> global 512 on 8 GPU) EXCEPT the optimizer block:
#   optimizer: HybridMuon, muon_lr=2e-3 (AdamW lr 2e-4 for biases/norms/embeddings).
#
# muon_lr ladder (N2L2C64): 0.02 hard-diverged, 0.01 soft-diverged (NaN cascade),
#   0.002 ran HEALTHY on the 2-card 8ep run (forces_mae 0.0396 @epoch5.6, no NaN).
# Stability levers carried in the config: warmup_epochs 0.5 / factor 1e-3. grad clip
# is left at 100 (it is near-inert for Muon -- Newton-Schulz normalizes the update,
# so muon_lr is the only real stability knob).
#
# Single-stage (direct only). Runs as an 8-proc torchrun; honors SENSECORE_* env
# vars if present (cluster GPU-pool job), else falls back to localhost:8.
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
# ~/.bashrc's `proxy_on` exports http(s)_proxy=http://127.0.0.1:45889 +
# all_proxy=socks5h://127.0.0.1:45889 -- a LOOPBACK tunnel only alive on the LOGIN
# machine. On a compute node 127.0.0.1:45889 is dead -> wandb's data-plane dies
# ("unexpected EOF"/Client.Timeout) so the run shows but every panel is empty.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem (no HybridMuon) ->
# `optimizer: HybridMuon` would crash with AttributeError. PYTHONPATH is searched
# before site-packages, so point it at this repo's src to override the image.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[direct] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
NGPU=8

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@15-bs@64x8_hybridmuon-mlr@2e-3-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='muon_direct_15ep_8card_mlr2e-3'
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

echo "[direct] ===== 8-card DIRECT 15ep ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[direct] ===== DONE. ckpt under $RUN_DIR/checkpoints/*-$DIRECT_ID/ ====="
