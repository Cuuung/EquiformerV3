#!/usr/bin/env bash
###############################################################################
# COMPILE-vs-EAGER net-effect ISOLATION (grad-only).
#
# The EAGER twin of the mstack-clean HIGHEST grad run. This run = EAGER + matmul
#   HIGHEST(fp32); the already-finished mstack-HIGHEST = compile + HIGHEST(fp32). They
#   differ in EXACTLY ONE config field (enable_compile True->False), share the SAME
#   bf16-b1.0 direct base, same HIGHEST precision, same moonshot optimizer, same 5ep.
#   => COMPILE isolated, precision HELD FIXED at fp32.
#
# WHY: the clean mstack triplet (2026-07-29) proved TF32 and grad-budget are BOTH
#   kappa-neutral at N2L2C64 (HIGHEST 0.4471 / HIGH-tf32 0.4448 / BUDGET0.8 0.4477).
#   That refuted "TF32 hurts kappa" and reassigned the old A0(compile+TF32) vs
#   E-G(eager+highest) +0.020 gap to COMPILE -- but A0/E-G ALSO differed in direct
#   budget, so compile was never cleanly isolated. THIS run isolates it:
#     this(eager+HIGHEST) vs mstack-HIGHEST(compile+HIGHEST), same b1.0 direct:
#       kappa match  -> compile kappa-neutral; the 30M keller loss is NOT compile
#                       (look to keller@30M-depth or the STABILIZED-70ep base).
#       eager better -> COMPILE itself costs kappa; maoruicong's in-model compile has a
#                       kappa price `highest` alone cannot recover at 30M.
#
# GRAD-ONLY: no direct stage. Loads the frozen mstack b1.0 direct best_checkpoint (pinned
#   in the yml AND passed on the CLI). Compare final MAE directly vs the HIGHEST twin
#   (forces_mae 0.04268, cos 0.6419); for kappa, run phonon/relax on this best_checkpoint.
#
# TOPOLOGY: 2 node x 4 GPU = 8 ranks. `--num-gpus` is PER NODE (fairchem gpus_per_node), hence 4/2.
#   grad bs 16 x 8 ranks x accum 4 = GLOBAL 512 (same as the HIGHEST twin -- comparison stays valid).
#
# NO EQV3_ACT_MEM_BUDGET: budget only affects the COMPILED backward; this run is EAGER,
#   so the env var is INERT and deliberately not set (setting it would falsely imply it matters).
# NO --amp (grad amp crashes Wigner); grad blocks are fp32.
#
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[compile-vs-eager] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/mstack_baseline/mstack-clean_N@2_L@2_C@64_gradft-5ep_moonshot_EAGER-highest-fp32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='mstack-clean_N2L2C64_gradft_5ep_moonshot_EAGER-highest-fp32'

# SAME b1.0 direct base as the HIGHEST twin -- mandatory for the isolation. Pinned in the yml too.
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-27-12-41-36-mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[compile-vs-eager] ERROR: mstack b1.0 direct base not found: $START_CKPT" >&2; exit 1
fi
echo "[compile-vs-eager] ===== EAGER twin ($ID) from mstack b1.0 direct base ====="
echo "[compile-vs-eager] direct base: $START_CKPT"

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

echo "[compile-vs-eager] ===== DONE. Compare vs the HIGHEST twin (compile+fp32, kappa 0.4471):"
echo "[compile-vs-eager]   MAE:   this(eager+HIGHEST) vs mstack grad forces_mae 0.04268 / cos 0.6419."
echo "[compile-vs-eager]   kappa: run phonon/relax on this best_checkpoint AND compare to 0.4471."
echo "[compile-vs-eager]   match => compile kappa-neutral (30M loss NOT compile);  eager better => compile costs kappa."
