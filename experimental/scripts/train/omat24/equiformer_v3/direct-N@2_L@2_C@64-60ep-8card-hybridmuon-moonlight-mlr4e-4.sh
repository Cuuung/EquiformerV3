#!/usr/bin/env bash
###############################################################################
# 8-GPU HybridMuon direct-force pretrain, N2L2C64, 60ep -- MOONLIGHT A/B test.
#
# PURPOSE: calibration check for the new Moonlight update scaling. Direct stage ONLY
# (no gradient-force stage 2): we just want the EARLY epochs to confirm whether the
# re-calibrated muon_lr=4e-4 is too aggressive before committing to a full run / gradft.
#
# A/B against the PROVEN ratio run (.../two_stage_pipelines/stage1-direct-...-mlr1e-3...):
#   that run used update_scale="ratio", muon_lr=1e-3, per-GPU 32 x 16 GPU = global 512.
# This run is IDENTICAL (model / data / loss e5-f10-s100 / 60ep / warmup 0.5 / clip 100 /
# momentum 0.95 / skip-step guard) EXCEPT:
#   (1) optimizer_params.update_scale: "ratio" -> "moonlight"
#       (scale 0.2*sqrt(max(rows,cols)) -> update RMS ~0.17*lr for EVERY matrix, shape-
#        decoupled; this is what makes muon_lr transferable across model sizes).
#   (2) optimizer_params.muon_lr: 1e-3 -> 4e-4 (RE-CALIBRATED; the two modes are on
#       different magnitude scales). 4e-4 matches the proven ratio-1e-3 run's TYPICAL
#       update RMS (~6.9e-5) and stays under the do-not-exceed bound ~6e-4 (where every
#       matrix would equal ratio-1e-3's worst matrix). FALLBACK: push toward the bound via
#       --optim.optimizer_params.muon_lr=6e-4 on the torchrun below.
#   (3) per-GPU batch 64 on 8 GPU (vs 32 on 16): global batch stays 512, so steps_per_epoch
#       and the cosine LambdaLR are bit-identical -> a STEP-FOR-STEP comparison.
#
# >>> LAUNCH ON 8 GPU <<< (set device count = 8 in the job). 64 x 8 = global 512.
#     On 16 GPU it would become global 1024 and the A/B (and schedule) no longer hold.
#
# WHAT TO WATCH (this is a calibration probe, not a stability proof):
#   - early 0-1ep: any "Found nans" / many "[HybridMuon] ... SKIPPED" lines => 4e-4 too hot,
#     drop to ~2e-4 and rerun. Moonlight raises the previously-quiet WIDE matrices ~2x, so
#     watch those first.
#   - overlay loss / "[HybridMuon] max NS-input RMS" against the ratio-1e-3 run to confirm
#     the trajectories track. (NOTE: 60ep metastable divergence only shows ~step 50k+, so a
#     few early epochs can REJECT a too-hot lr but cannot PROVE 60ep safety.)
#
# Single-stage (direct only). 8-proc torchrun; honors SENSECORE_* env if present
# (cluster GPU-pool job), else falls back to localhost:8.
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

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@60-bs@64x8_hybridmuon-moonlight-mlr@4e-4-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='muon_N2L2C64_direct_60ep_8card_moonlight_mlr4e-4'
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

echo "[direct] ===== 8-card DIRECT 60ep MOONLIGHT mlr4e-4 ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[direct] ===== DONE. ckpt under $RUN_DIR/checkpoints/*-$DIRECT_ID/ ====="
