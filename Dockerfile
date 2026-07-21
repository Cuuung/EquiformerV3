# EquiformerV3 torch.compile 训练镜像（对齐 eSEN 编译镜像范式）
#
# 与同事的 eSEN / UMA 镜像同范式（COPY packages/ + editable 安装；运行时从 AFS 挂载
# src/ 与 experimental/，改代码无需重打镜像）。针对 EquiformerV3 的差异：
#   * torch 2.11.0+cu128 —— 与 eSEN 一致，为守恒力 double-backward 的 dynamic=True
#     symbolic 编译路径（torch<2.11 会 ConstraintViolationError，每 shape 重编译）。
#     注意：这偏离 equiv3 env_setup.md 验证过的 torch 2.7.1，是当前 compile 分支的取舍。
#   * PyG 扩展只装 torch_scatter / torch_sparse / torch_cluster（pt211cu128 官方预编译）；
#     按本仓库结论 pyg_lib / torch_spline_conv 不被 import，省去。
#   * python3.10（ubuntu22.04 自带；与 eSEN 验证过的 cp310 pt211cu128 wheel 栈对齐）。
#   * timm==0.4.12 —— experimental/ 下的 EquiformerV3 模型需要，但不在 fairchem-core 依赖里。
#   * PYTHONPATH 同时含 src（fairchem 包）与仓库根（顶层 experimental 包 + my_main.py）。
#
# dynamic=True 本身是 yaml 运行时开关，镜像里不需要额外东西。
#
# Build: docker build -t equiformer_v3:compile .
# Run:   把 AFS 仓库挂到下面同一路径，CWD=仓库根后照常 torchrun my_main.py --config-yml ...
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# apt 换清华源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# pip 升级 + 清华为默认源
RUN python3 -m pip install --upgrade pip \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 1) torch 2.11.0+cu128（官方 whl 源；驱动需支持 CUDA 12.x）
RUN pip install --no-cache-dir torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2) PyG 扩展（pt211cu128 官方预编译；本仓库只 import 这三个，不装 pyg_lib / torch_spline_conv）
RUN pip install --no-cache-dir \
        torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3

# 只 COPY packages/（注册 fairchem CLI 入口 + 拉取非 torch 依赖；torch 已被上面装的
# 2.11.0+cu128 满足，pip 不会重装/降级）。packages/fairchem-core/src 是指向 ../../src 的
# 符号链接，hatch 的 fancy-pypi-readme / hatch-vcs hook 在 build 时按仓库根相对路径读 src 下
# 两个 fragment，故再单独 COPY 这两个文件到仓库根 src/；其余 src/ 与 experimental/ 不进镜像，
# 运行时从挂载的 AFS 读取（含本分支已提交的 compile 改动）。
COPY packages/ packages/
COPY src/fairchem/core/README.md src/fairchem/core/LICENSE.md src/fairchem/core/

# hatch-vcs 版本号：.git 不进镜像，用 PRETEND_VERSION 跳过 git-describe
ENV SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
# torchvision 0.26.0 与 torch 2.11.0 配套（cu128 源）。timm 0.4.12 顶层强制 import torchvision，
# 版本不配会报 "operator torchvision::nms does not exist"。装好后再 editable 安装 fairchem-core
# 并补装 experimental 需要但不在 fairchem-core 依赖里的 timm。
RUN pip install --no-cache-dir torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir -e "packages/fairchem-core" timm==0.4.12 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# ase-db-backends：packages/requirements.txt 列了它（git 版），但 editable 装 fairchem-core
# 只读 pyproject 依赖、不读 requirements.txt，故单独补装（DB dataset backend 运行时需要）。
RUN pip install --no-cache-dir ase-db-backends==0.10.0 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行时从挂载的 AFS 读取代码：src（fairchem 包）+ 仓库根（顶层 experimental 包 + my_main.py）
ENV PYTHONPATH=/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3/src:/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3

# 自检：只查能进镜像的部分（torch / pyg / timm 实包 + fairchem 入口元数据）。
# 注意：fairchem 真正代码运行时才从 AFS 挂载，build 期不在镜像里，故这里不 import fairchem。
RUN python3 -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.11'), 'need torch 2.11 for dynamic=True'" \
    && python3 -c "import torch_scatter, torch_sparse, torch_cluster; print('pyg stack OK')" \
    && python3 -c "import timm; print('timm', timm.__version__)" \
    && pip show fairchem-core | grep -E "Version|Editable|Location" \
    && which fairchem
