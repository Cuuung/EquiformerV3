#!/usr/bin/env bash
###############################################################################
# SINGLE-STAGE direct-force pretrain -- STABILIZED 70ep (16-GPU SenseCore job).
#   MODEL = paper's ~30M MPtrj model (Table 10): Lmax=4, 7 layers, C=128 (N7L4C128).
#   HybridMuon direct-force pretrain, 70ep, per-GPU 32 -> 16 GPU = global 512.
#   update_scale=moonlight, muon_lr=2e-4. --amp ON (fp16). NO stage-2 here.
#
# PURPOSE: the "train-it-full" headroom step toward SOTA -- undo diff #4 (direct 25 -> 70ep,
# the OFFICIAL horizon) WITHOUT re-triggering the deep-model post-convergence runaway that
# FORCED the 25ep early stop. This is NOT the safe production config; it is an EXPERIMENT.
#
# HOW IT IS STABILIZED (4 changes vs the 25ep base; see the yml header + docs §2.5):
#   (1) gamma WEIGHT DECAY [the real fix]: norm_weight_decay=1e-3 decays the RMSNorm/LayerNorm
#       gains (gamma), previously in the no-decay group. Bounds per-layer output RMS -> attacks
#       the mechanistic ROOT of Muon's runaway (Moonlight arxiv 2502.16982). *** NEW CODE PATH ***
#   (2) muon_lr RE-LIFT 1.5e-4 -> 2e-4: 2e-4 was healthy through ep20, only failed POST-convergence
#       (ep21) -- exactly what gamma-wd now guards.
#   (3) lr_min_factor 0.001 -> 0.01: relax the steep escape tail; a 70ep run must SUSTAIN lr, and
#       gamma-wd (not lr-decay-out) is now what prevents the runaway.
#   (4) max_epochs 25 -> 70.
#   SAFETY NET kept: spike_abs_threshold=4.0 aborts a runaway gamma-wd fails to hold ->
#       best_checkpoint = pre-runaway peak (a failed run still yields a usable ckpt).
#
# WATCH ep15-25 (old collapse band): if max NS-input RMS ramps / val cosine turns down, the
# escalation ladder is FIRST --optim.optimizer_params.norm_weight_decay=0.002 (cheapest),
# THEN --optim.optimizer_params.muon_lr=1.5e-4. Do NOT re-steepen the tail (recreates the 25ep cap).
#
# >>> LAUNCH ON 16 GPU <<< (set device count = 16 in the job). 32 x 16 = global 512.
#     On a different GPU count the global batch (and the muon_lr calibration) shift.
#
# MEMORY: direct checkpointing is OFF (fp16 + single backward; the 25ep run fit bs=32 on
#   A100-80GB). If it OOMs with DeNS, set gradient_checkpointing_block_list: [1,1,1,1,1,1,1]
#   in the DIRECT yml.
#
# Runs identically on every node; torchrun rendezvous via $MASTER_ADDR/$MASTER_PORT.
# GPU count is taken from $SENSECORE_ACCELERATE_DEVICE_COUNT (set 16 in the job).
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3
cd "$REPO"

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem. PYTHONPATH is searched
# before site-packages, so point it at this repo's src to override the image.
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Reduce CUDA fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sanity: the imported fairchem must be THIS working tree (HybridMuon) AND must carry the
# brand-new norm_weight_decay / norm_gain_names code path -- otherwise gamma-wd is SILENTLY
# skipped (old fairchem, or even commit f8b1ae4, lacks it) and the run is NOT stabilized.
python -c "import inspect, fairchem.core.trainers.base_trainer as bt, fairchem.core.common.muon as mu; \
src_bt=inspect.getsource(bt); src_mu=inspect.getsource(mu); \
assert 'HybridMuon' in src_bt, 'WRONG fairchem (no HybridMuon): '+bt.__file__; \
assert 'norm_weight_decay' in src_bt, 'fairchem too OLD: base_trainer lacks norm_weight_decay -> gamma-wd would be silently skipped: '+bt.__file__; \
assert 'norm_gain_names' in src_mu, 'fairchem too OLD: muon.py lacks norm_gain_names: '+mu.__file__; \
print('[stabilized] fairchem OK (norm_weight_decay path present):', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/maoruicong'

DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@70-bs@32x16_hybridmuon-moonlight-mlr@2e-4-alr@2e-4-wd@1e-3-normwd@1e-3-warmup@0.5-minf@0.01_dens-no-stress_loss-e5-f10-s100_STABILIZED.yml'
DIRECT_ID='muon_N7L4C128_direct_70ep_moonlight_mlr2e-4_normwd1e-3_STABILIZED'
##############################################################################

TORCHRUN_COMMON=(
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}"
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}"
  --master_addr   "${MASTER_ADDR:-127.0.0.1}"
  --master_port   "${MASTER_PORT:-29500}"
)
MAIN_COMMON=(
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-16}"
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}"
  --run-dir   "$RUN_DIR"
  --print-every 200 --seed 1
  --optim.num_workers=0
)

###############################################################################
# DIRECT (--amp off, use bf16 instead of fp16)
###############################################################################
echo "[stabilized] ===== DIRECT pretrain 70ep ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

echo "[stabilized] ===== DONE. Checkpoint under: $RUN_DIR/checkpoints/*-$DIRECT_ID/ ====="
