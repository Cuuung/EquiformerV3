#!/usr/bin/env bash
###############################################################################
# 2-GPU HybridMuon direct-force pretrain, N2L2C64, 8ep.
#
# Muon counterpart of the AdamW 8ep direct config (...epochs@8-bs@192x2...).
# Kept from it: loss 20:20:5, 8ep direct, denoising, use_compile.
# Stability levers carried over from the 2-card 2ep Muon smoke experience:
#   - muon_lr   -> 0.002   (further lowered: 0.01 diverged ~epoch1.7 on 8 GPU)
#   - warmup    epochs 0.5, factor 1e-3
#   - grad clip 100 -> 1.0
#   - batch     192 -> 32  (2 GPU -> global 64)
#
# Single-stage (direct only). Runs as a 2-proc single-node torchrun; honors
# SENSECORE_* env vars if present (cluster job), else falls back to localhost:2.
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
NGPU=2

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@8-bs@32x2_hybridmuon-mlr@2e-3-alr@2e-4-wd@1e-3-warmup@0.5-clip@1_dens-w@10-no-stress_loss-e20-f20-s5.yml'
DIRECT_ID='muon_direct_8ep_2card_mlr2e-3'
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

echo "[direct] ===== 2-card DIRECT 8ep ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[direct] ===== DONE. ckpt under $RUN_DIR/checkpoints/*-$DIRECT_ID/ ====="
