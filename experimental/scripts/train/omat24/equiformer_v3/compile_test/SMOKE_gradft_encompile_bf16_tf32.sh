#!/usr/bin/env bash
###############################################################################
# SMOKE (Tier 2) — CONSERVATIVE force (double-backward) · enable_compile
#   + compile_dynamic + bf16(model.use_amp) + tf32.
#   目的:在池上(maoruicong 训练镜像 torch 2.11+cu128)验证保守力 -dE/dx 双反向
#         在 dynamic=True symbolic 编译路径下跑通、能拿到梯度、不每 shape 重编译。
#   "启动→看头 1-2 条 print(每 50 步)→确认判据→停止作业"的冒烟,不是完整续训。
#   wandb 关闭(WANDB_MODE=disabled)。max_epochs=1 为保险。
#
# START_CKPT = N2L2C64 direct 15ep best(权重转移,backbone+energy_block shape-match)。
#   保守力 grad-ft 会覆盖 yml 里的 load_pretrained_weights → 用下面这个。
#
# PASS 判据(看第一条 print-every 日志):
#   - 编译通过;double-backward 能拿到力的梯度;loss 有限;step 递增;
#   - ckpt 加载日志正常(shape-match 部分加载可接受);不 OOM;不刷屏重编译。
# 一行运行:  bash "<本文件绝对路径>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export WANDB_SILENT=true

python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[smoke-gradft] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/gradient/compile_test/enable-compile_bf16_tf32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/smoke'
ID='N2L2C64_gradft_encompile_bf16_tf32_SMOKE'
START_CKPT='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/checkpoints/2026-06-13-04-01-04-muon_direct_15ep_8card_mlr2e-3/best_checkpoint.pt'

if [[ ! -s "$START_CKPT" ]]; then
  echo "[smoke-gradft] ERROR: start checkpoint not found or empty: $START_CKPT" >&2
  exit 1
fi
echo "[smoke-gradft] ===== Tier2 GRAD compile_dynamic+bf16+tf32 ($ID) from $START_CKPT ====="
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus  "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --run-dir   "$RUN_DIR" --identifier "$ID" \
  --print-every 50 --seed 1 \
  --optim.max_epochs=1 --optim.num_workers=0 \
  --optim.load_pretrained_weights="$START_CKPT"

echo "[smoke-gradft] ===== 若已看到 finite loss + 双反向梯度 + step 递增,可手动停止作业 ====="
