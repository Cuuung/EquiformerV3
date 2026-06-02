# EquiformerV3 training image (MPtrj reproduction) — China-network optimized.
#
# Versions match the validated local .venv: Python 3.11, torch 2.7.1 (cu126),
# torch_geometric 2.7.0, and PyG extensions torch_scatter 2.1.2 /
# torch_sparse 0.6.18 / torch_cluster 1.6.3. Only these three PyG extensions
# are imported by the codebase (torch_spline_conv / pyg_lib are NOT needed).
#
# Speed strategy (mirrors the user's reference Dockerfile):
#   * Tsinghua PyPI mirror for everything pip can get from PyPI.
#   * PyG extensions are COMPILED from Tsinghua source (avoids the slow,
#     overseas data.pyg.org wheel index).
#   * deadsnakes Python 3.11 so uv never downloads a Python from GitHub.
#   * torch itself still comes from the official cu126 index (unavoidable, and
#     guarantees the cu126 build matches the devel image's nvcc 12.6).
#
# Build:  docker build -t equiformer_v3:mptrj .
# Run:    docker run --gpus all --rm -it \
#           -v /path/to/mptrj_dataset:/data/mptrj -v /path/to/outputs:/outputs \
#           equiformer_v3:mptrj bash

# devel variant: includes nvcc + CUDA headers, required to compile torch_scatter.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System deps + Python 3.11 ─────────────────────────────────────────────────
# software-properties-common provides add-apt-repository; python3.11 comes from
# the deadsnakes PPA so we don't pull a standalone Python from GitHub via uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential software-properties-common \
        python3-pip libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ── uv via Tsinghua mirror (skips overseas files.pythonhosted.org) ────────────
RUN pip3 install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── venv on the system (deadsnakes) Python 3.11 ───────────────────────────────
RUN uv venv /opt/venv --python python3.11

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    # never let uv fetch a Python from GitHub; use the deadsnakes one above
    UV_PYTHON_PREFERENCE=only-system \
    # compile PyG extensions only for A100 (sm_80) to keep build time down
    TORCH_CUDA_ARCH_LIST="8.0" \
    # hatch-vcs reads the version from git; .git is excluded, so pin it
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev10

# ── PyTorch 2.7.1 + CUDA 12.6 (largest layer, cached early) ───────────────────
# From the official cu126 index to guarantee the cu126 build (matches nvcc 12.6).
RUN uv pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# ── Build tools (must be present before compiling PyG extensions) ─────────────
RUN uv pip install setuptools wheel packaging ninja \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Compile torch_scatter from Tsinghua source (~15 min) ──────────────────────
# --no-build-isolation: compile against the venv's already-installed torch so
#   setup.py can locate the CUDA toolkit and torch headers.
RUN uv pip install torch_scatter==2.1.2 \
    --no-binary torch_scatter --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Compile torch_sparse + torch_cluster from source (~25 min) ────────────────
RUN uv pip install torch_sparse==0.6.18 torch_cluster==1.6.3 \
    --no-binary torch_sparse --no-binary torch_cluster --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── torch_geometric (pure Python) ─────────────────────────────────────────────
RUN uv pip install torch_geometric==2.7.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Remaining pinned deps (upstream requirements file) ────────────────────────
COPY experimental/env/conda_requirements.txt /tmp/conda_requirements.txt
RUN uv pip install -r /tmp/conda_requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Extra runtime/util deps not in the pinned list ────────────────────────────
# typer for the eval CLIs, torchtnt (fairchem dep), huggingface_hub for
# checkpoints, httpx[socks] so HF works behind a SOCKS proxy.
RUN uv pip install \
        typer==0.25.1 torchtnt==0.2.4 ase_db_backends==0.10.0 \
        huggingface_hub==1.17.0 "httpx[socks]" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Project code + editable fairchem-core (after all compile layers, so code ──
#    edits don't trigger recompiles). packages/fairchem-core/src is a relative
#    symlink to ../../src, so the repo layout must be preserved (it is).
WORKDIR /workspace/equiformer_v3
COPY . /workspace/equiformer_v3
# --index-strategy unsafe-best-match + the pytorch extra-index keep PyPI's
# CPU-only torch from clobbering the installed cu126 build.
RUN uv pip install -e packages/fairchem-core \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match

# ── Sanity check: imports + experimental model registration ───────────────────
RUN python -c "import torch, torch_scatter, torch_sparse, torch_cluster; \
assert torch.version.cuda.startswith('12.6'), torch.version.cuda; \
from fairchem.core import OCPCalculator; \
from fairchem.core.common.utils import setup_imports; setup_imports(); \
from fairchem.core.common.registry import registry; \
assert 'equiformer_v3_dens' in registry.mapping['model_name_mapping']; \
print('fairchem + EquiformerV3 + PyG ext OK')"

CMD ["bash"]

# ── Optional: Matbench Discovery evaluation deps (not needed for training) ────
# RUN git clone https://github.com/janosh/matbench-discovery.git /opt/matbench-discovery \
#  && cd /opt/matbench-discovery && git checkout 375a8d6 \
#  && uv pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple \
#  && uv pip install moyopy pymatviz -i https://pypi.tuna.tsinghua.edu.cn/simple
#
# ── Note on image size ────────────────────────────────────────────────────────
# This is a single-stage build on a CUDA *devel* base (~large). To shrink the
# final image for upload, use a multi-stage build: compile here, then COPY
# /opt/venv into a `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` stage.
