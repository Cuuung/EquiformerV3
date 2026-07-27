#!/usr/bin/env bash
###############################################################################
# CLEAN maoruicong-STACK BASELINE on N2L2C64 -- the CORRECTED A0 (new baseline).
#
# Same recipe as dpa4-A0-baseline (moonshot: update_scale=moonlight, direct muon_lr 4e-4 /
#   gradft 1.5e-4, adamw 2e-4 / 5e-5, loss e5-f10-s100, wd 1e-3, max_atoms 150, DPA4 OFF)
#   PLUS maoruicong's stack (bf16 blocks on DIRECT, fp32 blocks on GRAD, enable_compile +
#   compile_dynamic), but with TWO deliberate corrections that make it a new baseline:
#
#   GOAL-1  budget misselection on DIRECT.  A0's script did a TOP-LEVEL `export
#           EQV3_ACT_MEM_BUDGET=0.6`, which leaked into the DIRECT torchrun (env is inherited by
#           every child) -> direct was forced to recompute activations -> low memory + ~15% slower,
#           so A0-direct never reproduced maoruicong's high-memory/fast direct profile. HERE the
#           export is REMOVED; budget is set ONLY on the grad torchrun line (its one legitimate use).
#           => DIRECT runs at the DEFAULT budget=1.0.  Direct config is REUSED byte-identically from
#           A0, so this run's direct differs from A0-direct in EXACTLY ONE thing: the budget env.
#           Measure it: direct steady-state s/epoch + peak mem here vs A0-direct (1946s, ~amp-low mem).
#
#   GOAL-2  precision on GRAD.  The grad config sets matmul_precision highest (pure fp32, NO TF32)
#           instead of A0's `high` (TF32), keeping compile ON. This isolates "compile alone" from
#           "TF32". Compare final kappa here vs A0 (0.6775, compile+TF32) and vs E-G (0.6579,
#           fp32-eager): if kappa(here) ~= E-G, compile itself is kappa-neutral and the +0.02 was
#           all TF32 -> production grad should be compile+highest (speed w/o the TF32 kappa cost).
#           ALSO record grad s/epoch: if compile+highest is not meaningfully faster than fp32-eager,
#           there is no reason to compile the grad stage at all (just use eager fp32).
#
# WHY NO REUSE OF A0's DIRECT CKPT: GOAL-1 is a SPEED/MEMORY measurement of budget=1 on direct --
#   it can ONLY be obtained by actually training direct at budget=1. Reusing A0's (budget-0.6) ckpt
#   would measure nothing for GOAL-1. Hence direct is retrained from scratch here.
#
# TOPOLOGY: 4 node x 2 GPU = 8 ranks.  `--num-gpus` is PER NODE (fairchem gpus_per_node), hence 2/4.
#   8 ranks preserves the global batch the configs assume: direct 64x8 = 512, grad 16x8x4 = 512
#   (same global batch as ablation-A / A0 -- only spread across 4 nodes; training math identical).
#
# NO --amp EVER: model.use_amp (bf16 blocks, geometry/Wigner fp32) is mutually exclusive with
#   optim.amp/fp16; direct uses use_amp True, grad uses use_amp False (grad amp crashes Wigner).
#
# LAUNCH (one line):  bash "<abs path to this file>"
#   RESUME direct-done:  DPA4_SKIP_DIRECT=1 bash "<this file>"   (or DPA4_DIRECT_CKPT=<path>)
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: EQV3_ACT_MEM_BUDGET is DELIBERATELY *NOT* exported here (that was A0's bug). Direct must run
#   at the default budget=1.0. The grad torchrun below sets it inline for that one command only.
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[mstack-clean] fairchem OK:', bt.__file__)"

RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
# DIRECT config is REUSED byte-identically from A0 (see GOAL-1 above): the ONLY difference from A0's
#   direct is the budget env, which this script no longer leaks. Do NOT fork a copy -- byte-identity
#   is what makes the budget comparison single-variable.
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/dpa4_ablation/dpa4_A0-baseline_N@2_L@2_C@64_direct-15ep_moonshot_bf16-compile.yml'
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/mstack_baseline/mstack-clean_N@2_L@2_C@64_gradft-5ep_moonshot_compile-HIGHEST-fp32.yml'
DIRECT_ID='mstack-clean_N2L2C64_direct_15ep_moonshot_bf16compile_BUDGET1'
GRADFT_ID='mstack-clean_N2L2C64_gradft_5ep_moonshot_compile_HIGHEST-fp32'

TORCHRUN=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-2}"
  --nnodes         "${SENSECORE_PYTORCH_NNODES:-4}"
  --node_rank      "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr    "${MASTER_ADDR:-127.0.0.1}"
  --master_port    "${MASTER_PORT:-29500}"
)
COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-2}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-4}"
  --run-dir   "$RUN_DIR" --print-every 200 --seed 1 --optim.num_workers=0
)

###############################################################################
# STAGE 1 -- DIRECT 15ep (bf16 blocks + compile + TF32-high) at DEFAULT budget=1.0 (GOAL-1).
###############################################################################
if [[ -n "${DPA4_SKIP_DIRECT:-}" || -n "${DPA4_DIRECT_CKPT:-}" ]]; then
  echo "[mstack-clean] ===== STAGE 1 SKIPPED (resume) ====="
else
  echo "[mstack-clean] ===== STAGE 1: DIRECT 15ep ($DIRECT_ID)  budget=DEFAULT(1.0) ====="
  torchrun "${TORCHRUN[@]}" my_main.py --mode train --config-yml "$DIRECT_CFG" \
    "${COMMON[@]}" --identifier "$DIRECT_ID"
fi

DIRECT_CKPT="${DPA4_DIRECT_CKPT:-$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -1)}"
if [[ ! -s "$DIRECT_CKPT" ]]; then
  echo "[mstack-clean] ERROR: stage-1 best_checkpoint not found (looked for *-$DIRECT_ID)" >&2; exit 1
fi
echo "[mstack-clean] stage-1 checkpoint: $DIRECT_CKPT"

###############################################################################
# STAGE 2 -- conservative-force GRAD-FT 5ep (fp32 blocks + compile + HIGHEST/no-TF32) (GOAL-2).
#   budget=0.6 set INLINE on THIS command ONLY -- its one legitimate use (grad double-backward mem).
#   budget is math-neutral (recompute vs store) so it does NOT affect kappa; here it just matches the
#   A0 grad memory setting, keeping the grad comparison single-variable (only matmul_precision differs).
###############################################################################
echo "[mstack-clean] ===== STAGE 2: GRAD-FT 5ep ($GRADFT_ID) from $DIRECT_CKPT  budget=0.6 (grad-only) ====="
EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}" \
torchrun "${TORCHRUN[@]}" my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${COMMON[@]}" --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$DIRECT_CKPT"

echo "[mstack-clean] ===== DONE. Next: phonon/relax on stage-2 best_checkpoint -> kappa_SRME + CPS."
echo "[mstack-clean]       GOAL-1: direct s/epoch + peak mem  vs A0-direct (1946s, low mem)."
echo "[mstack-clean]       GOAL-2: final kappa  vs A0 (0.6775, TF32+compile) and E-G (0.6579, fp32-eager);"
echo "[mstack-clean]               ALSO grad s/epoch: is compile+highest actually faster than fp32-eager?"
