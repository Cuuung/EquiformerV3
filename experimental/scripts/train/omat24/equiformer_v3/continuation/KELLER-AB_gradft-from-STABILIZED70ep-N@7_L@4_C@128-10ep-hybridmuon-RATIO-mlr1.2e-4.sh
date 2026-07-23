#!/usr/bin/env bash
###############################################################################
# STEP-1 KELLER-vs-moonshot A/B  ·  N7L4C128 conservative-force GRAD-FT (10ep)
#
# GOAL: does KELLER (update_scale=ratio) beat moonshot (update_scale=moonlight)
#   on kappa_SRME / CPS at scale, WITHOUT loss explosion in the 10ep double-backward?
#   This is the keller ARM. The moonshot arm = the existing gradft from the SAME
#   backbone (see CLARIFY note at bottom -- confirm it is matched before comparing).
#
# CLEAN A/B: this config differs from the moonshot gradft ONLY in
#     update_scale: moonlight -> ratio     (the scaling under test)
#     muon_lr:      5e-5       -> 1.2e-4    (matched EFFECTIVE step, not cranked)
#   Everything else identical: bs8/accum4 (global 512 on 16 GPU), clip_grad_norm 100,
#   spike_factor 8.0 / spike_abs 15.0, skip_nonfinite, warmup 0.1ep, gamma-wd NONE,
#   enable_compile+compile_dynamic+use_amp:False(fp32 blocks)+matmul high, 10ep.
#
# ANTI-EXPLOSION (keller is spike-prone: shape-dependent step, small matrices step big):
#   Guards are already in the yml. If epoch-1 shows growing SKIPPED / "Found nans" / loss
#   climbing, apply the LADDER (in order), re-launch:
#     (1) --optim.optimizer_params.spike_factor=6.0
#     (2) --optim.optimizer_params.muon_lr=8e-5
#     (3) --optim.clip_grad_norm=20
#
# LAUNCH: one line ->  bash "<abs path to this file>"   (16-GPU SenseCore job, image = maoruicong torch-2.11)
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

# strip dev-machine proxy so wandb egress works; point PYTHONPATH at our src (image bakes OLD fairchem).
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GRAD-STAGE ACCELERATION (maoruicong's tested-best): AOTAutograd min-cut partitioner
# activation_memory_budget = 0.6 -> trade recompute for memory, fits the conservative-force
# make_fx double-backward graph (cures the saved-tensor bloat / step-0 OOM). Default 1.0 = OOM-prone.
export EQV3_ACT_MEM_BUDGET="${EQV3_ACT_MEM_BUDGET:-0.6}"   # default 0.6; override by exporting before this bash call
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[keller-ab] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_CONT-from-STABILIZED70ep-bf16_epochs@10-bs@8x16x4-maxatoms150_hybridmuon-RATIO-mlr@1.2e-4-alr@5e-5-wd@1e-3_loss-e5-f10-s100_KELLER-AB.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'
ID='keller_N7L4C128_gradft_10ep_RATIO_mlr1.2e-4_from-STABILIZED70ep-bf16_KAPPA-AB'

# Shared backbone = "STABILIZED + maoruicong 混合精度 70ep" direct best_checkpoint --
# the SAME ckpt the moonshot arm (gradft-CONT-from-adamw-refine-...-mlr5e-5.sh) started from,
# so the kappa comparator is already produced (phonon: muon_n7l4c128_d70g10_from_adamw_refine_bf16).
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/maoruicong/checkpoints/2026-07-07-07-02-24-muon_N7L4C128_direct_70ep_moonlight_mlr2e-4_normwd1e-3_STABILIZED/best_checkpoint.pt'

if [[ ! -s "$START_CKPT" ]]; then
  echo "[keller-ab] ERROR: START_CKPT not set/found -> edit START_CKPT in this script: $START_CKPT" >&2
  exit 1
fi
echo "[keller-ab] ===== KELLER gradft 10ep ($ID) from $START_CKPT ====="
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 \
  --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[keller-ab] ===== DONE. Next: run phonon/relax on best_checkpoint.pt -> kappa_SRME + CPS vs moonshot arm ====="
