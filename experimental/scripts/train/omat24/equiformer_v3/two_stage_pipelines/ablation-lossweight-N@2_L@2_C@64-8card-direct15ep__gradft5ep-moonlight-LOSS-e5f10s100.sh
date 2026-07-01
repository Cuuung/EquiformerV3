#!/usr/bin/env bash
###############################################################################
# LOSS-WEIGHT ABLATION -- run A (BASELINE E:F:S = 5:10:100)
#
# Cheap N2L2C64 (~3M) proxy of the LATEST 30M (N7L4C128) Muon recipe, shortened to
# direct 15ep + grad-ft 5ep, on 8 GPU. This is the A/B PARTNER of:
#   ablation-lossweight-...-LOSS-e20f20s5.sh   (E:F:S = 20:20:5, paper config)
# The ONLY difference between the two scripts/configs is the loss weight.
#
#   MODEL    = N2L2C64 (2 layers, C=64, Lmax=2).
#   Stage-1  : HybridMuon MOONLIGHT direct pretrain, 15ep, bs=64/GPU -> 8 GPU = global 512.
#              update_scale=moonlight, muon_lr=4e-4, alr=2e-4. --amp ON (fp16).
#   Stage-2  : HybridMuon MOONLIGHT gradient-force finetune, 5ep, bs=16/GPU + grad_accum 4
#              (= global 512). update_scale=moonlight, muon_lr=1.5e-4, alr=5e-5,
#              max_atoms=150. fp32, NO --amp (amp crashes the Wigner build in so3.py).
#
# Optimizer config is MATCHED to the most recent 30M run (moonlight + full divergence
# guards incl. the absolute backstop spike_abs_threshold=4.0). The one knob that is NOT
# carried over verbatim is muon_lr: the small model is calibrated to moonlight 4e-4
# (direct) -- the 30M's 1.5e-4 was depth-lowered and would UNDERTRAIN N2L2C64 in 15ep.
# grad-ft muon_lr = 4e-4 x ~0.33 = 1.5e-4, mirroring the 30M's direct->gradft ratio.
#
# EXPECTED WALL CLOCK (8 GPU): direct ~6.3h (25 min/ep, measured on the 06-17 8-card
# moonlight run) + grad-ft ~7h (3.4x the direct per-epoch cost) ~= ~13-14h total.
#
# >>> LAUNCH ON 8 GPU <<< (set device count = 8 in the job). On a different GPU count the
#     global batches (and the muon_lr calibration) shift -- keep it at 8.
#   NOTE: launch AFTER the old job is stopped. fixed muon.py + the configs are read at
#   startup via PYTHONPATH=src.
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem (no HybridMuon).
# PYTHONPATH is searched before site-packages -> point it at this repo's src.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[abl-A] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

# --- Stage 1: MOONLIGHT direct pretrain (--amp ON), loss 5:10:100 ------------
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_epochs@15-bs@64x8_hybridmuon-moonlight-mlr@4e-4-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100_ablation.yml'
DIRECT_ID='abl_N2L2C64_direct_15ep_moonlight_mlr4e-4_loss-e5f10s100'

# --- Stage 2: MOONLIGHT grad-ft finetune (fp32, NO --amp), loss 5:10:100 -----
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@2_L@2_C@64_pt-dens-ft-no-reg_hybridmuon-moonlight-mlr@1.5e-4-alr@5e-5-epochs@5-bs@16x8x4-maxatoms150-wd@1e-3_loss-e5-f10-s100_ablation.yml'
GRADFT_ID='abl_N2L2C64_gradft_5ep_moonlight_mlr1.5e-4_maxatoms150_bs16x8x4_loss-e5f10s100'
##############################################################################

TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
MAIN_COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1
  --optim.num_workers=0
)

###############################################################################
# Stage 1: DIRECT  (--amp ON)
###############################################################################
echo "[abl-A] ===== Stage 1: MOONLIGHT DIRECT pretrain 15ep ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[abl-A] Stage 1 finished. Locating its best_checkpoint.pt ..."
CKPT=$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT:-}" || ! -s "$CKPT" ]]; then
  echo "[abl-A] ERROR: could not find a non-empty best_checkpoint.pt for identifier '$DIRECT_ID'." >&2
  exit 1
fi
echo "[abl-A] Using pretrained weights: $CKPT"

###############################################################################
# Stage 2: GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[abl-A] ===== Stage 2: MOONLIGHT GRAD-FT finetune 5ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[abl-A] ===== DONE (loss 5:10:100). Stage-1 ckpt: $CKPT ====="
