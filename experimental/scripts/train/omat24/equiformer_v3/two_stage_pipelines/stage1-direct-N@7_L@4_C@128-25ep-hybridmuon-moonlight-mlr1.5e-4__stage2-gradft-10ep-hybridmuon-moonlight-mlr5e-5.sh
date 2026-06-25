#!/usr/bin/env bash
###############################################################################
# Two-stage chained pipeline -- MOONLIGHT, PAPER ~30M model (intended for a 16-GPU SenseCore job):
#   MODEL = paper's ~30M MPtrj model (Table 10): Lmax=4, 7 layers, C=128 (N7L4C128).
#   Stage-1: HybridMuon direct-force pretrain, 25ep, per-GPU 32
#            -> 16 GPU = global 512. update_scale=moonlight, muon_lr=1.5e-4. --amp ON (fp16).
#   Stage-2: HybridMuon gradient-force finetune, 10ep, per-GPU 16 + grad_accum 2 (= global 512),
#            gradient_checkpointing=[1]x7. update_scale=moonlight, muon_lr=5e-5. fp32, NO --amp.
#
# PURPOSE: paper comparison on the real 30M model. Vs the original paper recipe, ONLY two
# axes differ -- (A) optimizer AdamW -> HybridMuon (moonlight), (B) epochs direct 70 -> 25.
# Everything else follows the user's established recipe (loss e5-f10-s100, warmup 0.5,
# dropout 0.1/0.05, DeNS, global batch 512), NOT reverted to paper values.
#
# FOURTH attempt -- THREE prior failures, now addressed by LR + schedule + an absolute backstop:
#   (1) 40ep/mlr@4e-4: silent STALL from a guard BUG (EMA frozen on skip -> permanent ratchet).
#       FIXED in muon.py (skips ADVANCE the EMA + spike_max_consecutive_skips force-commit).
#   (2) 35ep/mlr@4e-4 (06-19): DIVERGED at ep9.6 (network-wide spike, AMP scaler collapsed).
#       muon_lr too hot for this deep model (moonlight transfers across WIDTH, not DEPTH).
#   (3) 35ep/mlr@2e-4 (06-20): HEALTHY to ep20 (val cosine 0.756, still rising) then a SLOW
#       self-amplifying runaway from ep21 (max NS-input RMS 0.12->9.4 over ~0.5ep). Never NaN ->
#       no crash -> ran to ep35 LOBOTOMIZED (val flat 9.96 for 11ep). Two causes: Muon's update
#       is gradient-magnitude-INDEPENDENT (near a converged sharp min it keeps constant-size
#       steps and wanders out), and the EMA-relative guard SELF-DISARMS in a slow runaway.
#   => FIX (this run): (a) muon_lr 2e-4 -> 1.5e-4; (b) Stage-1 epochs 35 -> 25 + lr_min_factor
#      0.01 -> 0.001 (front-load Muon, then DECAY OUT before the post-convergence runaway);
#      (c) NEW absolute backstop in muon.py (spike_abs_threshold=4.0) aborts a sustained runaway
#      the relative guard can't catch -- best_checkpoint = pre-runaway peak, so abort loses nothing.
#      NOTE: launch AFTER the old job is stopped (fixed muon.py + new lr/schedule read at startup
#      via PYTHONPATH src). Watch ep15-22 (failure (3)'s band): rising max NS-input RMS = detune
#      to --optim.optimizer_params.muon_lr=1e-4.
#
# >>> LAUNCH ON 16 GPU <<< (set device count = 16 in the job).
#     Stage-1 32x16 = global 512 ; Stage-2 16x16x2 (bs x GPU x accum) = global 512.
#     On a different GPU count the global batches (and the muon_lr calibration) shift.
#
# WHAT TO WATCH:
#   - Stage-1 ep15-22 (the danger band): a rising max NS-input RMS or falling val cosine = the
#     runaway is starting. The absolute backstop will ABORT it (best_checkpoint = peak); if it
#     fires, resume from that ckpt at --optim.optimizer_params.muon_lr=1e-4.
#   - Stage-1 memory: direct checkpointing is OFF; if it OOMs on A100-80GB with DeNS, set
#     gradient_checkpointing_block_list: [1,1,1,1,1,1,1] in the DIRECT yml.
#   - Stage-2: bs already dropped to 16 + grad_accum 2 (= global 512) to fix the bs=32 OOM
#     (conservative-force double-backward + fp32). If it STILL OOMs, drop to bs=8 + grad_accum 4.
#
# Chaining:
#   1. Stage-1 runs with a fixed --identifier ($DIRECT_ID). fairchem writes its
#      checkpoint to $RUN_DIR/checkpoints/<timestamp>-$DIRECT_ID/best_checkpoint.pt
#   2. After Stage-1's torchrun returns 0, we glob the newest match on $DIRECT_ID
#      and feed best_checkpoint.pt to Stage-2 via the CLI override
#      --optim.load_pretrained_weights=<ckpt> (overrides the yml's placeholder).
#   3. Stage-1 keeps --amp; Stage-2 must NOT use --amp (amp crashes Wigner build
#      in conservative-force mode -- gradient force is fp32-only).
#
# Runs identically on every node; torchrun rendezvous via $MASTER_ADDR/$MASTER_PORT
# coordinates ranks, and the checkpoint lives on shared storage (all nodes see it).
# GPU count is taken from $SENSECORE_ACCELERATE_DEVICE_COUNT (set 16 in the job).
###############################################################################
set -euo pipefail

cd /mnt/afs/home/yaolekai/MLIP/equiformer_v3

# --- wandb egress fix: strip any dev-machine proxy leaked from the submit shell --
# ~/.bashrc's `proxy_on` exports http(s)_proxy=http://127.0.0.1:45889 and
# all_proxy=socks5h://127.0.0.1:45889 -- a LOOPBACK tunnel that only exists on the
# LOGIN machine. On a compute node 127.0.0.1:45889 is dead -> wandb's data-plane dies
# ("unexpected EOF"/Client.Timeout) so the run shows but every panel is empty.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# DO NOT pip install: the docker image bakes in an OLD fairchem (no HybridMuon) ->
# `optimizer: HybridMuon` would crash with AttributeError. PYTHONPATH is searched
# before site-packages, so point it at this repo's src to override the image.
export PYTHONPATH=/mnt/afs/home/yaolekai/MLIP/equiformer_v3/src:${PYTHONPATH:-}
# Reduce CUDA fragmentation (helps the memory-heavy fp32 conservative-force grad-ft stage).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem imported (no HybridMuon branch): '+bt.__file__; print('[pipeline] fairchem OK:', bt.__file__)"

############################## EDIT THESE #####################################
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai'

# --- Stage 1: HybridMuon MOONLIGHT direct-force pretrain (--amp ON) -----------
# update_scale=moonlight, muon_lr=1.5e-4, 25ep. global batch 32x16=512. PAPER 30M model.
DIRECT_CFG='experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@7_L@4_C@128_rbf@10_attn-grid@14-8_ffn-grid@14_merge-ln_epochs@25-bs@32x16_hybridmuon-moonlight-mlr@1.5e-4-alr@2e-4-wd@1e-3-warmup@0.5_dens-no-stress_loss-e5-f10-s100.yml'
DIRECT_ID='muon_N7L4C128_direct_25ep_moonlight_mlr1.5e-4'

# --- Stage 2: HybridMuon MOONLIGHT gradient-force finetune (fp32, NO --amp) ---
# update_scale=moonlight, muon_lr=5e-5, 10ep, global batch 16x16x2=512 (checkpointing=[1]x7).
GRADFT_CFG='experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_C@128_attn-hidden@32_rbf@10_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_pt-dens-ft-no-reg_hybridmuon-moonlight-mlr@5e-5-alr@5e-5-epochs@10-bs@16x16x2-wd@1e-3_loss-e5-f10-s100.yml'
GRADFT_ID='muon_N7L4C128_gradft_10ep_moonlight_mlr5e-5'
##############################################################################

# Honor SenseCore rendezvous when present (cluster GPU-pool job); else fall back to a
# single local node so the same script can be launched with a plain `bash <script>`.
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
# Stage 1: DIRECT  (--amp ON)
###############################################################################
echo "[two-stage] ===== Stage 1: MOONLIGHT DIRECT pretrain 25ep ($DIRECT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$DIRECT_CFG" --amp \
  "${MAIN_COMMON[@]}" \
  --identifier "$DIRECT_ID"

###############################################################################
# Locate the checkpoint Stage 1 just produced
###############################################################################
echo "[two-stage] Stage 1 finished. Locating its best_checkpoint.pt ..."
CKPT=$(ls -dt "$RUN_DIR"/checkpoints/*-"$DIRECT_ID"/best_checkpoint.pt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT:-}" || ! -s "$CKPT" ]]; then
  echo "[two-stage] ERROR: could not find a non-empty best_checkpoint.pt for identifier '$DIRECT_ID'." >&2
  echo "[two-stage] Looked under: $RUN_DIR/checkpoints/*-$DIRECT_ID/best_checkpoint.pt" >&2
  exit 1
fi
echo "[two-stage] Using pretrained weights: $CKPT"

###############################################################################
# Stage 2: GRAD-FINETUNE  (fp32, NO --amp; overrides yml's load_pretrained_weights)
###############################################################################
echo "[two-stage] ===== Stage 2: MOONLIGHT GRAD-FT finetune 10ep ($GRADFT_ID) ====="
torchrun "${TORCHRUN_COMMON[@]}" \
  my_main.py --mode train --config-yml "$GRADFT_CFG" \
  "${MAIN_COMMON[@]}" \
  --identifier "$GRADFT_ID" \
  --optim.load_pretrained_weights="$CKPT"

echo "[two-stage] ===== DONE. Stage-1 ckpt: $CKPT ====="
