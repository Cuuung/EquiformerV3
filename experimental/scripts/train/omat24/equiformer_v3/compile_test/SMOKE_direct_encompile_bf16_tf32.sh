#!/usr/bin/env bash
###############################################################################
# SMOKE (Tier 1) — DIRECT force · enable_compile + bf16(model.use_amp) + tf32.
#   目的:在池上(maoruicong 训练镜像 torch 2.11+cu128)验证 compile 层行为一致。
#   这是"启动→看头 1-2 条 print(每 50 步)→确认判据→停止作业"的冒烟,不是完整训练。
#   wandb 关闭(WANDB_MODE=disabled):冒烟不记录。max_epochs=1 为保险(忘停也会自然结束)。
#
# PASS 判据(看第一条 print-every 日志):
#   - 编译无 ConstraintViolationError;loss 有限(非 NaN/inf);step 递增;不 OOM。
# 一行运行:  bash "<本文件绝对路径>"
###############################################################################
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/equiformer_v3
cd "$REPO"

# 剥离从提交 shell 泄漏的开发机代理(否则 wandb/egress 会串)。
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# 镜像里烘的是旧 fairchem(无 HybridMuon);PYTHONPATH 先于 site-packages → 指向本仓库 src。
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 冒烟:彻底关掉 wandb(no-op,零网络零记录)。
export WANDB_MODE=disabled
export WANDB_SILENT=true

# 自检:确认导入的是本仓库带 HybridMuon 的 fairchem,而非镜像里的旧版。
python -c "import inspect, fairchem.core.trainers.base_trainer as bt; assert 'HybridMuon' in inspect.getsource(bt), 'WRONG fairchem (no HybridMuon): '+bt.__file__; print('[smoke-direct] fairchem OK:', bt.__file__)"

CFG='experimental/configs/omat24/mptrj/experiments/direct/compile_test/enable-compile_bf16_tf32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/yaolekai/smoke'
ID='N2L2C64_direct_encompile_bf16_tf32_SMOKE'

echo "[smoke-direct] ===== Tier1 DIRECT compile+bf16+tf32 ($ID) ====="
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
  --optim.max_epochs=1 --optim.num_workers=0

echo "[smoke-direct] ===== 若已看到 finite loss + step 递增,可手动停止作业 ====="
