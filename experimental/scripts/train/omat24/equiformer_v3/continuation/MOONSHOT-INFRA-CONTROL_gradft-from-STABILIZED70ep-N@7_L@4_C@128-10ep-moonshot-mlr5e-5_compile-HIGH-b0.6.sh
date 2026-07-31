#!/usr/bin/env bash
###############################################################################
# 30M MOONSHOT + maoruicong-INFRA control group (grad-only, N7L4C128, 10ep).
#
# PURPOSE: cleanly SPLIT the keller-30M ablation (kappa 0.3086) into "optimizer" vs
#   "infra" at 30M scale. That run = keller(ratio 1.2e-4) + compile+HIGH(TF32)+budget0.6,
#   from the STABILIZED-70ep base. The flagship (kappa 0.2764) = moonshot + EAGER-fp32,
#   SAME base. So keller-vs-flagship bundled TWO changes (optimizer AND infra).
#
#   THIS run = moonshot (FLAGSHIP optimizer) + compile+HIGH+budget0.6 (KELLER's infra),
#   SAME STABILIZED-70ep base. It is the missing corner that separates them:
#     this vs flagship (moonshot, eager-fp32) : isolates the maoruicong compile+HIGH+budget
#         STACK effect on kappa at 30M (optimizer & base held fixed = moonshot / STABILIZED-70ep).
#     this vs keller  (keller,  compile+HIGH+budget0.6) : isolates the OPTIMIZER (moonshot vs
#         keller) at 30M under identical infra.
#   Together with the flagship + keller numbers this fully decomposes the 0.3086.
#
# CONFIG: copied byte-for-byte from the KELLER-AB 30M grad yml; the ONLY changes are the two
#   optimizer fields (update_scale ratio->moonlight, muon_lr 1.2e-4->5e-5) -- restoring the
#   flagship optimizer (verified identical wd/betas/spike-guards/AdamW-lr from its checkpoint).
#   Infra UNCHANGED: enable_compile True + compile_dynamic True + matmul_precision high(TF32) +
#   use_amp False (fp32 blocks) + EQV3_ACT_MEM_BUDGET 0.6.
#
# DIRECT BASE: maoruicong STABILIZED-70ep (moonlight mlr2e-4, normwd1e-3) -- the SAME base as
#   BOTH the flagship and the keller run. Pinned in the yml AND passed on the CLI.
#
# TOPOLOGY: 2 node x 8 GPU = 16 ranks. `--num-gpus` is PER NODE (fairchem gpus_per_node) = 8.
#   grad bs 8 x 16 ranks x accum 4 = GLOBAL 512 (same as flagship & keller -- comparison valid).
# budget=0.6 matters here (compile ON). NO --amp (grad amp crashes Wigner); blocks are fp32.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"   # compile ON -> budget active; 0.6 matches keller & A0
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[moonshot-infra-control] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/eqv3_gradft_N@7_L@4_C@128_10ep_MOONSHOT-mlr5e-5_compile-HIGH-b0.6_from-STABILIZED70ep_INFRA-CONTROL.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='moonshot-infra-control_N7L4C128_gradft_10ep_moonshot_mlr5e-5_compile-HIGH-b0.6_from-STABILIZED70ep'

# SAME STABILIZED-70ep base as the flagship AND the keller run -- mandatory for both isolations.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/maoruicong/checkpoints/2026-07-07-07-02-24-muon_N7L4C128_direct_70ep_moonlight_mlr2e-4_normwd1e-3_STABILIZED/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[moonshot-infra-control] ERROR: STABILIZED-70ep base not found: $START_CKPT" >&2; exit 1
fi
echo "[moonshot-infra-control] ===== moonshot + compile+HIGH+budget0.6 ($ID) ====="
echo "[moonshot-infra-control] direct base: $START_CKPT"
echo "[moonshot-infra-control] budget=$EQV3_ACT_MEM_BUDGET  matmul=high(TF32)  compile=ON  optimizer=moonshot(mlr5e-5)"

torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-2}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-2}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[moonshot-infra-control] ===== DONE. Decomposes the keller-30M 0.3086:"
echo "[moonshot-infra-control]   vs flagship (moonshot eager-fp32, kappa 0.2764) => maoruicong compile+HIGH+budget net kappa effect at 30M."
echo "[moonshot-infra-control]   vs keller   (keller  compile+HIGH+budget0.6, kappa 0.3086) => optimizer (moonshot vs keller) at 30M."
