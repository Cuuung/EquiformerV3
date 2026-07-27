#!/usr/bin/env bash
###############################################################################
# DPA4 ABLATION on N2L2C64 -- arm A2-D2F2: D2 (cross-focus competition, F=2)
#
# Separate row on purpose: D2 targets EXPRESSIVITY, not cutoff smoothness -- opposite
#   direction from D1/D4, so packaging them would make the result uninterpretable.
#   CAVEAT for reading the result: softmax forces sum_F alpha = 1, so turning competition on
#   scales total message amplitude down by ~1/F. That is part of DPA4's mechanism, not a bug,
#   but a positive/negative D2 result cannot be attributed to 'competition' alone.
#
# RECIPE: identical to the moonshot N2L2C64 15ep+5ep pair behind kappa_SRME 0.4638 / CPS 0.7302
#   (docs/KELLER_VS_MOONSHOT_N2L2C64.md): HybridMuon moonshot (update_scale=moonlight),
#   direct 15ep muon_lr 4e-4 / adamw 2e-4, grad-ft 5ep muon_lr 1.5e-4 / adamw 5e-5,
#   loss e5-f10-s100, wd 1e-3, max_atoms 150.
#   PLUS maoruicong's stack: enable_compile + compile_dynamic, bf16 blocks on DIRECT
#   (use_amp True), fp32 blocks on GRAD-FT (use_amp False, matching the production 30M
#   recipe), matmul_precision=high (TF32), and EQV3_ACT_MEM_BUDGET for the grad stage.
#
# TWO STAGES RUN BACK TO BACK. Stage 2 loads stage 1's best_checkpoint.
#   The DPA4 switches are set IDENTICALLY in both stages -- mandatory, see below.
#
# !! WHY "identically" IS MANDATORY !!
#   `load_pretrained_weights` (equiformer_v3_dens_trainer.py) builds its dict FROM the model and
#   only overwrites keys the checkpoint happens to contain, then calls load_state_dict on that
#   same dict. The key sets are equal by construction, so strict=True CAN NEVER FIRE: a parameter
#   present in the model but absent from the checkpoint is silently left at random init, with no
#   log line at all. D1's z_bias_raw and D2's focus_compete.* are exactly such parameters.
#   The transfer check below fails the job loudly instead.
#
# LAUNCH (one line):  bash "<abs path to this file>"
#   TOPOLOGY: 2 node x 4 GPU = 8 ranks. `--num-gpus` is PER NODE (fairchem maps it to
#   gpus_per_node, my_main.py:71), hence 4 / 2 -- NOT 8 / 2. 8 ranks is what the configs
#   assume: direct bs 64x8 and grad-ft bs 16x8x4 both give the SAME global batch as the
#   moonshot baseline this arm is compared against. Changing the rank count changes the
#   global batch and silently breaks that comparison.
#   On SenseCore the SENSECORE_* env vars override these defaults automatically.
#   Override the grad-stage activation-memory budget by exporting EQV3_ACT_MEM_BUDGET first.
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Grad-stage acceleration (maoruicong's tested best): AOTAutograd min-cut partitioner budget.
export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[dpa4-A2-D2F2] fairchem OK:', bt.__file__)"

ARM='A2-D2F2'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/dpa4_ablation/dpa4_A2-D2F2_N@2_L@2_C@64_direct-15ep_moonshot_bf16-compile.yml'
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/dpa4_ablation/dpa4_A2-D2F2_N@2_L@2_C@64_gradft-5ep_moonshot_bf16-compile.yml'
DIRECT_ID="dpa4_${ARM}_N2L2C64_direct_15ep_moonshot_bf16compile"
GRADFT_ID="dpa4_${ARM}_N2L2C64_gradft_5ep_moonshot_compile"

TORCHRUN=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}"
  --nnodes         "${SENSECORE_PYTORCH_NNODES:-2}"
  --node_rank      "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr    "${MASTER_ADDR:-127.0.0.1}"
  --master_port    "${MASTER_PORT:-29500}"
)
COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-2}"
  --run-dir   "$RUN_DIR" --print-every 200 --seed 1 --optim.num_workers=0
)

###############################################################################
# STAGE 1 -- DIRECT 15ep (bf16 blocks + compile + TF32).  NOTE: no --amp, ever:
#   model.use_amp (block-scoped bf16, geometry fp32) is mutually exclusive with optim.amp/fp16.
###############################################################################
# RESUME: if the direct stage already finished (e.g. only stage 2 failed), skip straight to it:
#     DPA4_SKIP_DIRECT=1 bash "<this file>"
# It picks the newest matching stage-1 run dir. Set DPA4_DIRECT_CKPT=<path> to pin a specific one.
if [[ -n "${DPA4_SKIP_DIRECT:-}" || -n "${DPA4_DIRECT_CKPT:-}" ]]; then
  echo "[dpa4-$ARM] ===== STAGE 1 SKIPPED (resume) ====="
else
  echo "[dpa4-$ARM] ===== STAGE 1: DIRECT 15ep ($DIRECT_ID) ====="
  torchrun "${TORCHRUN[@]}" my_main.py --mode train --config-yml "$DIRECT_CFG" \
    "${COMMON[@]}" --identifier "$DIRECT_ID"
fi

DIRECT_CKPT="${DPA4_DIRECT_CKPT:-$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -1)}"
if [[ ! -s "$DIRECT_CKPT" ]]; then
  echo "[dpa4-$ARM] ERROR: stage-1 best_checkpoint not found (looked for *-$DIRECT_ID)" >&2; exit 1
fi
echo "[dpa4-$ARM] stage-1 checkpoint: $DIRECT_CKPT"

###############################################################################
# GUARD -- every DPA4 parameter the stage-2 model needs must exist in the stage-1 checkpoint.
#   Without this the run would train a half-random model and finish silently (see header).
###############################################################################
python - "$GRADFT_CFG" "$DIRECT_CKPT" <<'PYEOF'
import sys, yaml, torch
from fairchem.core.common.registry import registry
import fairchem.experimental.models.equiformer_v3.equiformer_v3          # noqa: F401  (register)
import fairchem.experimental.models.equiformer_v3.equiformer_v3_dens     # noqa: F401
from fairchem.experimental.models.equiformer_v3.dpa4_ops import (
    check_dpa4_switch_transfer, dpa4_parameter_names)

cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
mcfg = dict(yaml.safe_load(open(cfg_path))["model"])
mcfg.pop("enable_compile", None); mcfg.pop("compile_dynamic", None)   # construction-only check
cls = registry.get_model_class(mcfg.pop("name"))
model = cls(**mcfg)
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
missing = check_dpa4_switch_transfer(model, sd)
have = dpa4_parameter_names(model)
print(f"[dpa4-guard] stage-2 DPA4 params: {len(have)}  not supplied by stage-1 ckpt: {len(missing)}")
if missing:
    print("[dpa4-guard] FAIL -- these would silently stay at RANDOM INIT:", file=sys.stderr)
    for m in missing:
        print("    " + m, file=sys.stderr)
    print("[dpa4-guard] the two stages were NOT switched identically. Aborting.", file=sys.stderr)
    sys.exit(1)
print("[dpa4-guard] OK: stage-1 -> stage-2 DPA4 parameter transfer is complete.")
PYEOF

###############################################################################
# STAGE 2 -- conservative-force GRAD-FT 5ep (fp32 blocks + compile + TF32 + mem budget)
###############################################################################
echo "[dpa4-$ARM] ===== STAGE 2: GRAD-FT 5ep ($GRADFT_ID) from $DIRECT_CKPT ====="
torchrun "${TORCHRUN[@]}" my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${COMMON[@]}" --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$DIRECT_CKPT"

echo "[dpa4-$ARM] ===== DONE. Next: phonon/relax on stage-2 best_checkpoint -> kappa_SRME + CPS."
echo "[dpa4-$ARM]       Compare A1/A2 against the A0 arm (NOT against the old 0.4638: different precision stack)."
