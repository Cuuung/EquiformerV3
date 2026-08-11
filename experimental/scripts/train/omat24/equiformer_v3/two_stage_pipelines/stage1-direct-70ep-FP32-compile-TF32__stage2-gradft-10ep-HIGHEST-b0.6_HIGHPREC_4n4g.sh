#!/usr/bin/env bash
###############################################################################
# 30M HIGHEST-FEASIBLE-PRECISION two-stage run (N7L4C128), 4 node x 8 GPU = 32 GPU.
#
# GOAL: run the model at the highest precision that is currently FEASIBLE.
#   Stage-1 DIRECT (70ep): compile + TF32 + **bf16 OFF** (pure fp32 blocks), budget=1(default).
#     -> higher precision than the flagship's bf16 direct (drops the 16-bit block autocast),
#        keeps TF32+compile for speed. Pure fp32 forward; NO --amp on the CLI.
#   Stage-2 GRAD (10ep): compile + **highest**(no TF32) + budget=0.6, moonshot mlr5e-5.
#     -> highest-precision grad matmul (drops TF32) while keeping compile+budget for memory.
#
# MEMORY: fp32 direct activations ~2x bf16. We halve per-GPU batch 32->16 so the fp32-bs16 activation
#   footprint == the proven bf16-bs32 one (4B*16 == 2B*32). Global batch stays 512 by DOUBLING GPUs
#   (16->32) instead of accum: direct 16/GPU x 32 GPU x accum1 = 512; grad 8/GPU x 32 GPU x accum2 = 512.
#   gradient_checkpointing stays OFF (compile+checkpointing is an untested combo here).
#   If step-0 OOMs, drop DIRECT batch_size 16->8. If it fits with headroom, raise to 24/32.
#
# BUDGET split (per stage, NOT a global export):
#   direct torchrun -> EQV3_ACT_MEM_BUDGET=1   (user spec: default 1; compile ON so it applies)
#   grad   torchrun -> EQV3_ACT_MEM_BUDGET=0.6 (memory headroom for the fp32 double backward)
#
# TOPOLOGY: 4 node x 8 GPU = 32 ranks. --num-gpus is PER NODE (fairchem gpus_per_node) = 8.
#
# HANDOFF: Stage-1 writes $RUN_DIR/checkpoints/<ts>-$DIRECT_ID/best_checkpoint.pt; we glob the
#   newest match and feed it to Stage-2 via --optim.load_pretrained_weights.
#
# NOTE (kappa expectation): our 30M data show the maoruicong grad infra bundle (compile+HIGH+
#   budget0.6) cost +0.0265 kappa vs the EAGER flagship; this run keeps compile+budget but drops
#   TF32 (highest), so it tests whether dropping TF32 recovers that. If pure max-kappa is the goal
#   (not max-precision-with-compile), an EAGER-highest grad would be even cleaner (the flagship ran
#   eager grad fine at 30M) -- but this recipe follows the requested compile+highest+budget0.6.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[highprec] fairchem OK:', bt.__file__)"

RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/eqv3_direct_N@7_L@4_C@128_70ep_FP32blocks_compile-TF32_budget1_HIGHPREC.yml'
DIRECT_ID='highprec_N7L4C128_direct_70ep_FP32blocks_compile-TF32_budget1'

GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/eqv3_gradft_N@7_L@4_C@128_10ep_MOONSHOT_compile-HIGHEST_b0.6_HIGHPREC.yml'
GRADFT_ID='highprec_N7L4C128_gradft_10ep_moonshot_compile-HIGHEST_budget0.6_from-FP32-TF32-direct'

TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-4}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
COMMON_ARGS=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-4}"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1 --optim.num_workers=0
)

# ---------------------------------------------------------------------------
# Stage 1: DIRECT  (pure fp32 blocks + TF32 + compile; NO --amp; budget=1)
# ---------------------------------------------------------------------------
echo "[highprec] ===== Stage 1: DIRECT 70ep ($DIRECT_ID)  fp32-blocks+TF32+compile, budget=1 ====="
EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-1}" \
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" \
  "${COMMON_ARGS[@]}" \
  --identifier "$DIRECT_ID"

echo "[highprec] Stage 1 finished. Locating its best_checkpoint.pt ..."
CKPT=$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT:-}" || ! -s "$CKPT" ]]; then
  echo "[highprec] ERROR: no best_checkpoint.pt for identifier '$DIRECT_ID' under $RUN_DIR/checkpoints/*-$DIRECT_ID/" >&2
  exit 1
fi
echo "[highprec] Stage-1 direct base: $CKPT"

# ---------------------------------------------------------------------------
# Stage 2: GRAD-FT  (compile + highest(no TF32) + budget=0.6, moonshot mlr5e-5)
# ---------------------------------------------------------------------------
echo "[highprec] ===== Stage 2: GRAD-FT 10ep ($GRADFT_ID)  compile+HIGHEST(fp32), budget=0.6 ====="
EQV3_ACT_MEM_BUDGET=0.6 \
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${COMMON_ARGS[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[highprec] ===== DONE. direct base: $CKPT  ; grad id: $GRADFT_ID ====="
echo "[highprec] compare kappa/CPS vs flagship (0.2764 / 0.8346全量) and infra-control (0.3029)."
