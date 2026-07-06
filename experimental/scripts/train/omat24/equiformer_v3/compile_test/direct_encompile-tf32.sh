#!/usr/bin/env bash
# N2L2C64 direct 15ep, 自建 enable_compile(+compile_dynamic) + tf32(yml), 无 amp/bf16. 单阶段。
set -euo pipefail

REPO=/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

CFG='experimental/configs/omat24/mptrj/experiments/direct/compile_test/enable-compile_tf32.yml'
RUN_DIR='/mnt/afs/share/checkpoint/equiformerV3/maoruicong'   # EDIT: 输出目录
ID='N2L2C64_direct_encompile_tf32'

torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29500}" \
  my_main.py --mode train --config-yml "$CFG" \
  --num-gpus "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --num-nodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --run-dir "$RUN_DIR" --identifier "$ID" \
  --print-every 200 --seed 1 --optim.num_workers=0
