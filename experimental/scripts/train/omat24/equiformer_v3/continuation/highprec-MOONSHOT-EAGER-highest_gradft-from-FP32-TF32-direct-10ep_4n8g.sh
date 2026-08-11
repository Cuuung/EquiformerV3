#!/usr/bin/env bash
###############################################################################
# 30M EAGER-highest gradft (N7L4C128), from the highprec FP32-blocks direct base.
# 4 node x 8 GPU = 32 GPU.
#
# PURPOSE: the FLAGSHIP-STYLE EAGER twin of the highprec compile-HIGHEST run.
#   highprec compile arm (kappa 0.2812): direct FP32-blocks+compile+TF32, grad COMPILE+HIGHEST+b0.6.
#   THIS arm (eager):                    direct SAME base,          grad EAGER(HIGHEST).
#   The ONLY differences vs the highprec gradft run are:
#     - enable_compile True -> False (compile OFF -> pure eager)
#     - compile_dynamic True -> False
#     - NO EQV3_ACT_MEM_BUDGET (budget only affects compiled backward; eager inert)
#   Precision HELD: matmul=highest(no TF32), use_amp=False(fp32 blocks), same moonshot mlr5e-5.
#   => clean isolation of GRAD STAGE COMPILE at 30M, on the SAME FP32 direct base.
#
# EXPECTATION (from our N2L2C64 and 30M data):
#   - N2L2C64: compile κ-neutral (eager 0.4457 vs compile 0.4471, −0.0014).
#   - 30M: highprec compile+HIGHEST+b0.6 = 0.2812 (vs flagship-eager 0.2764 the gap was
#     +0.0048 including a DIFFERENT direct base). This run removes compile → if compile
#     alone costs something at 30M, this should be BETTER than 0.2812 (closer to 0.2764);
#     if it's neutral, κ stays ≈0.2812 → the remaining +0.0048 gap vs flagship IS the
#     direct-base change (bf16 STABILIZED vs fp32+TF32). Either way = clean compile-kappa
#     at 30M on the NEW FP32 direct base.
#
# GRAD-ONLY: loads the highprec FP32-blocks direct best_checkpoint.pt (pinned in yml AND
#   passed on the CLI). NO direct stage here.
#
# TOPOLOGY: 4 node x 8 GPU = 32 ranks. --num-gpus PER NODE = 8.
#   grad bs 8 x 32 ranks x accum2 = GLOBAL 512 (same as highprec compile arm & flagship).
# NO EQV3_ACT_MEM_BUDGET (eager -> inert).
# NO --amp (grad amp crashes Wigner); blocks are fp32.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[eager-highprec] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/eqv3_gradft_N@7_L@4_C@128_10ep_MOONSHOT_EAGER-highest_from-FP32-TF32-direct_HIGHPREC.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='highprec_N7L4C128_gradft_10ep_moonshot_EAGER-highest_from-FP32-TF32-direct'

# SAME FP32-blocks direct base as the highprec compile-HIGHEST run. Pinned in yml too.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-08-05-00-40-32-highprec_N7L4C128_direct_70ep_FP32blocks_compile-TF32_budget1/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[eager-highprec] ERROR: FP32 direct base not found: $START_CKPT" >&2; exit 1
fi
echo "[eager-highprec] ===== EAGER twin ($ID) ====="
echo "[eager-highprec] direct base: $START_CKPT"
echo "[eager-highprec] grad: EAGER(no compile) + highest(fp32 no TF32) + moonshot mlr5e-5 + 10ep"
echo "[eager-highprec] NO budget (eager -> inert). Compare vs highprec compile arm (kappa 0.2812)."

torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-4}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-4}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[eager-highprec] ===== DONE. Compare:"
echo "[eager-highprec]   highprec compile arm (kappa 0.2812) vs THIS eager arm -> compile κ-cost at 30M (same FP32 direct)."
echo "[eager-highprec]   flagship eager          (kappa 0.2764) vs THIS eager arm -> direct-base change (bf16→fp32 net effect)."
