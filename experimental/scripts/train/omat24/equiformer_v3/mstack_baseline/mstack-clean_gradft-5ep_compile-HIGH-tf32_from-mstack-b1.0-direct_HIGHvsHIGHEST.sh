#!/usr/bin/env bash
###############################################################################
# PURE high-vs-highest ISOLATION (grad-only).
#
# The TF32 twin of the mstack-clean grad run. This run = compile + matmul_precision HIGH (TF32);
#   the already-finished mstack-clean grad = compile + HIGHEST (fp32). They differ in EXACTLY ONE
#   config field (matmul_precision), share the SAME mstack b1.0 direct base, same compile ON,
#   same budget 0.6, same optimizer. => TF32 isolated with compile HELD ON.
#
# WHY: we never had a clean TF32 ablation. A0 grad = compile+high(TF32); E-G grad = eager+highest;
#   that pair bundled compile WITH precision, so the +0.02 kappa gap (A0 0.6775 vs E-G 0.6579)
#   could not be pinned on TF32 vs compile. This twin separates them:
#     this(compile+HIGH) vs mstack(compile+HIGHEST):
#       MAE/kappa match  -> TF32 harmless; the A0/E-G +0.02 was compile.
#       this worse       -> TF32 is the culprit; compile exonerated.
#
# GRAD-ONLY: no direct stage. Loads the frozen mstack b1.0 direct best_checkpoint (pinned in the
#   yml AND passed on the CLI). Compare final MAE directly vs the HIGHEST twin (forces_mae 0.04268,
#   cos 0.6419); for kappa, run phonon/relax on both stage-2 checkpoints.
#
# TOPOLOGY: 2 node x 4 GPU = 8 ranks. `--num-gpus` is PER NODE (fairchem gpus_per_node), hence 4/2.
#   grad bs 16 x 8 ranks x accum 4 = GLOBAL 512 (same as the HIGHEST twin -- comparison stays valid).
#
# budget=0.6 set INLINE on the torchrun (compiled backward needs it); math-neutral, matches the twin.
# NO --amp (grad amp crashes Wigner); grad blocks are fp32, TF32 only touches fp32 matmuls.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[high-vs-highest] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/mstack_baseline/mstack-clean_N@2_L@2_C@64_gradft-5ep_moonshot_compile-HIGH-tf32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGH-tf32'

# SAME b1.0 direct base as the HIGHEST twin -- mandatory for the isolation. Pinned in the yml too.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-27-12-41-36-mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[high-vs-highest] ERROR: mstack b1.0 direct base not found: $START_CKPT" >&2; exit 1
fi
echo "[high-vs-highest] ===== TF32 twin ($ID) from mstack b1.0 direct base ====="
echo "[high-vs-highest] direct base: $START_CKPT"

EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}" \
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-2}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-2}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[high-vs-highest] ===== DONE. Compare vs the HIGHEST twin (compile+fp32):"
echo "[high-vs-highest]   MAE:   this(compile+HIGH) vs mstack grad forces_mae 0.04268 / cos 0.6419."
echo "[high-vs-highest]   kappa: run phonon/relax on this best_checkpoint AND the HIGHEST twin's."
echo "[high-vs-highest]   match => TF32 harmless (A0/E-G +0.02 was compile);  worse => TF32 is the culprit."
