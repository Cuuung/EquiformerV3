# Environment Record — EquiformerV3 MPtrj reproduction

> Purpose: a **record** of the environment that successfully ran EquiformerV3
> (MPtrj) training/inference, so we have a known-good baseline to fall back on.
> This is documentation only — not a portability/migration guide.

Last updated: 2026-06-02 · Baseline repo commit: `a7300c5`

---

## 1. Known-good baseline (fall back to this if something breaks)

| Item | Value |
|------|-------|
| Repo commit | `a7300c5` (branch `env_base`) |
| Docker image | `equiformer_v3:mptrj` (local), built from the repo `Dockerfile` |
| Validated | single-point inference + MPtrj training smoke (both on GPU, inside the image) |
| Smallest verified config | `experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_N@2_L@2_C@64_...epochs@1...no-stress.yml` (Lmax=2, 2 blocks, C=64, 4.76M params) |

---

## 2. Host / hardware (the machine where it was built & validated)

| Item | Value |
|------|-------|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 5.15.0-113-generic |
| CPU cores | 128 |
| GPU | NVIDIA A100-SXM4-80GB ×8 |
| NVIDIA driver | 550.90.07 |
| CUDA (driver-supported) | 12.x (driver 550 → up to CUDA 12.4 base; 12.6 runtime works via minor-version compat) |

Cluster training target: **A100 (sm_80)** — matches this host's GPU arch.

---

## 3. Python environment

- **Python**: 3.11.15
- **Built with**: [`uv`](https://github.com/astral-sh/uv) (`uv venv` + `uv pip install`).
  Note: the venv has **no `pip` module** — use `uv pip ...` for any change.
- **fairchem-core**: editable install of this repo (`pip install -e packages/fairchem-core`,
  whose `src` is a relative symlink to the repo `src/` → resolves to `src/fairchem`).
- Experimental models/trainers register via `src/fairchem/experimental/.include`
  (`./models`, `./trainers`) during `setup_imports()`.

### Core ML stack (pinned)

| Package | Version |
|---------|---------|
| torch | 2.7.1+cu126 |
| torchvision / torchaudio | 0.22.1+cu126 / 2.7.1+cu126 |
| CUDA build (torch) | 12.6 · cuDNN 9.5 · NCCL 2.26.2 (all bundled in the torch wheels) |
| torch-geometric | 2.7.0 |
| torch_scatter / torch_sparse / torch_cluster | 2.1.2 / 0.6.18 / 1.6.3 (all `+pt27cu126`) |
| torch_spline_conv / pyg-lib | 1.2.2 / 0.5.0 (`+pt27cu126`; not imported by the model) |
| e3nn | 0.5.6 |
| numpy / scipy | 2.2.6 / 1.16.1 |
| ase / pymatgen | 3.25.0 / 2025.6.14 |
| numba | 0.61.2 |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 |
| wandb / tensorboard | 0.21.0 / 2.20.0 |

Full pinned list: see [Appendix A](#appendix-a--full-pinned-package-list) (139 packages).

---

## 4. Docker image configuration

Defined by the repo `Dockerfile` (China-network optimized). Summary:

| Aspect | Choice |
|--------|--------|
| Base image | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` (**devel** = has `nvcc`, needed to compile PyG ext) |
| Python | 3.11 from **deadsnakes PPA** (`UV_PYTHON_PREFERENCE=only-system` so uv never downloads from GitHub) |
| Package manager | `uv` (installed via Tsinghua mirror) |
| PyPI mirror | Tsinghua (`pypi.tuna.tsinghua.edu.cn`) for everything pip can fetch |
| torch | `torch==2.7.1` from official `download.pytorch.org/whl/cu126` (unavoidable; matches nvcc 12.6) |
| PyG extensions | **compiled from Tsinghua source** (`torch_scatter` 2.1.2, `torch_sparse` 0.6.18, `torch_cluster` 1.6.3) with `TORCH_CUDA_ARCH_LIST="8.0"` (A100) |
| fairchem-core | `uv pip install -e packages/fairchem-core` with `--extra-index-url` pytorch + `--index-strategy unsafe-best-match` (prevents CPU-only torch from clobbering the cu126 build) |
| Version stamp | `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev10` (`.git` is excluded from the image) |
| Image size | ~24.3 GB (single-stage on a devel base) |

Build:
```bash
docker build -t equiformer_v3:mptrj .
```

Run (GPU; mount dataset + outputs, weights/data are NOT baked in — see `.dockerignore`):
```bash
docker run --gpus all --rm -it \
  -v /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb:/data:ro \
  -v /path/to/outputs:/outputs \
  equiformer_v3:mptrj bash
```

Notes:
- `docker build` does **not** use the host's localhost SOCKS proxy; the torch step
  needs direct reach to `download.pytorch.org` (or a docker daemon proxy config).
- PyG-extension compile takes ~40 min (it is CPU-bound).

---

## 5. Datasets (not in git; mount at runtime)

| Dataset | Path on host |
|---------|--------------|
| MPtrj aselmdb | `/mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/{train,val}` (train: 320 shards + `metadata.npz`; val: + `metadata.npz`) |

Checkpoints: downloaded from HF `mirror-physics/equiformer_v3` into `checkpoints/`
(git-ignored). MPtrj-only model = `checkpoint/mptrj_gradient.pt`.

---

## 6. Recreating the environment (quick reference)

The authoritative recipe is the `Dockerfile`. For a bare-metal venv, mirror
`experimental/docs/env_setup.md` but pin to the versions above (cu126, not cu128),
or regenerate the lock with:
```bash
uv pip freeze > experimental/env/requirements.lock.txt
```

---

## Appendix A — full pinned package list

<details><summary>139 packages (uv pip freeze, 2026-06-02)</summary>

```text
ase==3.25.0
ase_db_backends==0.10.0
e3nn==0.5.6
fairchem-core==0.1.dev10+ga7300c58d
hydra-core==1.3.2
huggingface_hub==1.17.0
httpx==0.28.1
lmdb==1.7.3
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-nccl-cu12==2.26.2
omegaconf==2.3.0
pandas==2.3.1
pydantic==2.11.7
pyg-lib==0.5.0+pt27cu126
pymatgen==2025.6.14
scipy==1.16.1
spglib==2.7.0
submitit==1.5.3
sympy==1.14.0
tensorboard==2.20.0
timm==0.4.12
torch==2.7.1+cu126
torch-geometric==2.7.0
torch_cluster==1.6.3+pt27cu126
torch_scatter==2.1.2+pt27cu126
torch_sparse==0.6.18+pt27cu126
torch_spline_conv==1.2.2+pt27cu126
torchaudio==2.7.1+cu126
torchtnt==0.2.4
torchvision==0.22.1+cu126
triton==3.3.1
typer==0.25.1
wandb==0.21.0
```

> Abbreviated to the ML-relevant packages. To dump the complete 139-package set
> at any time: `uv pip freeze`. (CUDA `nvidia-*` wheels are pulled automatically
> as torch dependencies.)

</details>
