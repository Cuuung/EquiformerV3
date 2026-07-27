#!/usr/bin/env bash
###############################################################################
# E-G  ·  GRAD-STAGE PRECISION ISOLATION  (does the conservative-stage acceleration hurt kappa?)
#
# QUESTION (the one open item in porting maoruicong's stack): direct-stage mixed precision is
#   accepted as good (maoruicong's fine-grained bf16 > eqV3's naive AMP); grad-stage compile
#   speeds training up; the ONLY unverified risk is whether the GRAD-stage precision changes
#   (TF32 matmul + in-model compile on the -dE/dx double-backward) degrade the EVAL metrics.
#   kappa_SRME reads FC3 -- a derivative of exactly that path -- so it is the sensitive probe.
#
# DESIGN (single variable = grad-ft numerical stack):
#   * START from the EXISTING A0 bf16 direct ckpt -- direct is NOT retrained (reuse the 8.7h).
#   * Run ONLY the 5ep conservative grad-ft, with the grad stage reverted to PURE fp32 EAGER:
#       matmul_precision high->highest (TF32 off) + enable_compile True->False.
#   * Everything else byte-identical to the A0 grad-ft (moonshot mlr 1.5e-4, 5ep, loss e5f10s100,
#     bs 16x8x4, max_atoms 150, use_amp already False, DPA4 switches all baseline).
#
# READOUT (send the resulting best_checkpoint to phonon/relax -> kappa_SRME):
#   * kappa -> ~0.46  => the grad-stage TF32/compile WAS the eval killer. Direct bf16 exonerated.
#                        Production fix: keep maoruicong's direct bf16 (fast), run the conservative
#                        stage fp32-eager for kappa-critical models.
#   * kappa stays ~0.67 => the damage is baked into the bf16 DIRECT basin (5ep fp32 grad-ft can't
#                        repair it); the direct stage needs an fp32 retrain (the report's E1).
#   Compare against: A0 grad-ft (TF32+compile) kappa 0.6775, and the old fp32 ablation kappa 0.4638.
#
# TOPOLOGY: 2 node x 4 GPU = 8 ranks (--num-gpus is PER NODE; SENSECORE_* override on the pool).
# LAUNCH (one line):  bash "<abs path to this file>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Mem budget is a memory/recompute knob, not a precision one -- keep it for the double-backward.
export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[E-G] fairchem OK:', bt.__file__)"

RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/dpa4_ablation/dpa4_A0-baseline_N@2_L@2_C@64_gradft-5ep_moonshot_GRADFP32-eager.yml'
GRADFT_ID='EG_A0direct-bf16_gradft_5ep_moonshot_FP32-eager'

# Pin the EXISTING A0 direct bf16 checkpoint (identical basin as the A0 grad-ft comparator started from).
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-07-23-05-30-40-dpa4_A0-baseline_N2L2C64_direct_15ep_moonshot_bf16compile/best_checkpoint.pt'
if [[ ! -s "$START_CKPT" ]]; then
  echo "[E-G] ERROR: A0 direct bf16 ckpt not found: $START_CKPT" >&2; exit 1
fi
echo "[E-G] ===== grad-ft 5ep PURE-FP32-EAGER ($GRADFT_ID) from A0 direct bf16 ====="
echo "[E-G] start ckpt: $START_CKPT"

torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-2}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-2}" \
  --run-dir   "$RUN_DIR" --identifier "$GRADFT_ID" \
  --print-every 200 --seed 1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[E-G] ===== DONE. Send best_checkpoint.pt to phonon/relax -> kappa_SRME."
echo "[E-G]       kappa ~0.46 => grad-stage precision was the killer; ~0.67 => direct bf16 basin is."
