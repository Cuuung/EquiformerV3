# compile_dev — EquiformerV3 torch.compile Gate Suite

This directory contains the regression gates for the EquiformerV3 `torch.compile`
integration (branch `compile-equiformer-v3`, Tasks 0-6).  Every gate exits with
code 0 on PASS and 1 on FAIL.

---

## Environment

All gates must be run via the compile env wrapper — it activates the esen
`torch-2.11` venv and sets `PYTHONPATH` to expose `fairchem.core` from this repo:

```bash
bash scripts/compile_env.sh compile_dev/<gate>.py [args]
```

Plain `python compile_dev/<gate>.py` will NOT find `fairchem.core` and will fail.

Venv path: `/mnt/afs/home/maoruicong/esen/.venv-torch211`

---

## Gate Scripts

### 1. `verify_rotation_migration.py`

**Purpose:** Verify that the edge-rotation migration from Gram-Schmidt + e3nn
(old path) to UMA Euler angles (new path, Task 1) is:
- Rotationally equivariant (energy invariant, forces equivariant).
- Roll-invariant (output stable across random gamma seeds).
- Numerically lossless vs the old path (within fp32 noise floor).

**Pass criteria:**
- `energy_invariance_err < 1e-5`
- `force_equivariance_err < 1e-3`
- Losslessness vs old Gram-Schmidt baseline: `energy_err < max(1e-5, 5×baseline_roll_e)`,
  `force_err < max(1e-3, 5×baseline_roll_f)`.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py
```

**Baseline files:** `compile_dev/_rot_baseline.pt` — recorded from the old
Gram-Schmidt + e3nn path BEFORE the Task-1 migration.  To re-record:
```bash
bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py --baseline
```
Note: `--baseline` must be run against PRE-migration code (old Gram-Schmidt path
in the model).  The current codebase has the new Euler path; running `--baseline`
now would record the new path, defeating the comparison.  The `.pt` file is
git-ignored (dev artifact).

---

### 2. `verify_core_compute_refactor.py`

**Purpose:** Confirm that extracting `core_compute` (Task 2 / Task 3 refactor)
is numerically lossless vs the original monolithic forward for both the base
(`EquiformerV3_OC`) and DeNS (`EquiformerV3DeNS_OC`) models.

**Pass criteria:** Max abs diff < 1e-6 for energy, forces, and stress on both
model variants.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py
```

**Baseline files:** `compile_dev/_core_baseline.pt` — recorded from the
pre-refactor monolithic forward.  To re-record (must be run against code BEFORE
the `core_compute` extraction):
```bash
bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py --record
```
The `.pt` file is git-ignored.

---

### 3. `test_stage_direct.py`

**Purpose:** Stage 3 gate — verify that `enable_compile=True` (direct-force path,
`plain_compile` wrapper on `core_compute`) is numerically equivalent to eager
(`enable_compile=False`) on both base and DeNS variants.  Also asserts
`_compiled_core is not None` (no silent eager fallback).

**Pass criteria:**
- `max_diff(energy) < 1e-5`
- `max_diff(forces) < 1e-5`
- `_compiled_core` attribute is set (not None) on the compiled model.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_stage_direct.py
```

---

### 4. `test_stage_conservative_base.py`

**Purpose:** Task-4 hard gate (base / non-DeNS) — verify that the conservative
`_forward_gradient` path (double-backward, `make_fx` CompiledForceRegion) produces
gradients aligned with eager on every trainable parameter.

**Pass criteria (gradient alignment — NOT loss trajectories):**
- Concatenated gradient cosine similarity: `cos(g_eager, g_compiled) > 0.999`
- Global norm ratio: `|1 - ||g_compiled|| / ||g_eager||| < 0.05` (5%)
- Compiled grad-param set ⊇ eager grad-param set (strip_detach check: no severing)
- All meaningful params (`|g| > 1e-4 × max_param_grad_norm`) get gradient
- Changed system shape re-traces without crash, with `cos > 0.999`

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_stage_conservative_base.py
```

> **Note:** This gate runs `make_fx` + inductor double-backward compilation.
> First run can take **several minutes**.

---

### 5. `test_stage_conservative_dens.py`

**Purpose:** Same as `test_stage_conservative_base.py` but for `EquiformerV3DeNS_OC`
with a denoising-active batch (mixed noise mask, stress live).  Exercises the
force-embedding (`SO3Linear`) path and the eager `dens_block` alongside the
compiled `core_compute` region.

**Pass criteria:** Same as gate 4, applied to DeNS model with denoising batch.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_stage_conservative_dens.py
```

> **Note:** Same compilation overhead as gate 4.  Expect several minutes.

---

### 6. `test_mutual_exclusion.py`

**Purpose:** Task-5 gate — verify the trainer mutex between `optim.use_compile`
(outer `torch.compile`) and `model.enable_compile` (inner `plain_compile`):
1. `use_compile=True, enable_compile=False` → outer `torch.compile` IS invoked.
2. `enable_compile=True, use_compile=False` → outer NOT invoked; `optimize_ddp=False`.
3. Both True → `ValueError` raised.

**Pass criteria:** All 3 cases behave as specified.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_mutual_exclusion.py
```

---

### 7. `test_stage_dynamic_base.py`

**Purpose:** Dynamic=True gate (base / non-DeNS) — verify the distinguishing
property of `compile_dynamic=True` over bucketing: 3 distinct system sizes
(different natoms AND nedges) share ONE compiled kernel. The CompiledForceRegion
cache stays at size 1 throughout.

Also checks force/stress are finite and numerically consistent with eager
(tolerance <1e-5), and shapes 2+ run in <2s (zero recompile; first compile ~27s).

**Pass criteria:**
- `cache_size == 1` for all 3 shapes (single_kernel=True)
- `max_diff(forces) < 1e-5`, `max_diff(stress) < 1e-5`
- All outputs finite
- Shapes 2+ complete in <2s (no recompile)

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_stage_dynamic_base.py
```

**Requirement:** `torch >= 2.11` for dynamic symbolic `make_fx`. Skips with code 0
on older torch.

---

### 8. `test_stage_dynamic_dens.py`

**Purpose:** Same as `test_stage_dynamic_base.py` but for `EquiformerV3DeNS_OC`
with denoising-active batch. Verifies single-kernel dynamic=True behavior holds
with DeNS force-embedding path and eager `dens_block`.

**Pass criteria:** Same as gate 7, applied to DeNS model with denoising batch.

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_stage_dynamic_dens.py
```

**Requirement:** `torch >= 2.11` — same as gate 7.

---

### 9. `test_single_frame.py`

**Purpose:** Finding-A gate — verify that backbone and heads use the SAME random
SO(2) frame (single-frame semantics). After core_compute refactor returns edge
features, both paths share identical edge_distance_exp and edge_envelope_weight.

**Pass criteria:**
- Edge features from core_compute match those used by heads (bit-exact equality)
- No redundant `_forward_edge` calls causing frame desync

**Rerun command:**
```bash
bash scripts/compile_env.sh compile_dev/test_single_frame.py
```

---

## Full Regression Run

```bash
for s in verify_rotation_migration verify_core_compute_refactor test_stage_direct \
          test_stage_conservative_base test_stage_conservative_dens test_mutual_exclusion \
          test_stage_dynamic_base test_stage_dynamic_dens test_single_frame; do
  echo "=== $s ==="
  bash scripts/compile_env.sh compile_dev/$s.py || echo "FAIL $s"
done
```

Expected: all 9 gates PASS.

---

## In-training profiler (`EQV3_PROF`)

Env-gated `torch.profiler` hook inside `equiformer_v3_dens_trainer.py` (mirrors
esen's `ESEN_PROF`). Default off / no-op; master-rank only. Profiles the **real
training loop** (real batches, DDP, optimizer, compile+precision path) — use it
to see which CUDA kernels dominate a step (e.g. whether the `reduce_edge` scatter
fuses under bf16, roadmap §8.3-A).

```bash
export EQV3_PROF=1          # main switch (ESEN_PROF=1 also accepted)
export EQV3_PROF_WAIT=5     # steps skipped entirely — waits out MODEL warmup (compile/cudagraph/cache)
export EQV3_PROF_WARMUP=5   # steps traced-then-discarded — waits out the PROFILER's own startup cost
export EQV3_PROF_ACTIVE=20  # steps actually recorded (the sampling window)
export EQV3_PROF_DIR=./eqv3_prof   # tensorboard trace output dir
```

Then launch training normally. One cycle = `wait → warmup → active` (default
5+5+20 steps). After the active window, the log prints the top-25 CUDA kernels
(`key_averages().table(sort_by="cuda_time_total")`) and a trace is written to
`EQV3_PROF_DIR`. To A/B a change: run once with it, `git stash` the change and run
again, compare the `indexFunc*` / triton-scatter kernel times.

---

## Untracked Baseline `.pt` Files

`compile_dev/_rot_baseline.pt` and `compile_dev/_core_baseline.pt` are
**intentionally git-ignored dev artifacts**.  They were recorded during
development against PRE-refactor code:

| File | When to record | Command |
|------|---------------|---------|
| `_rot_baseline.pt` | Against code with old Gram-Schmidt path (before Task-1 migration) | `bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py --baseline` |
| `_core_baseline.pt` | Against code before `core_compute` extraction (before Task-2/3 refactor) | `bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py --record` |

On a fresh checkout the baselines will be absent.  Gates 1 and 2 will skip the
losslessness comparison and print a warning.  To re-establish the baselines, you
must check out a pre-refactor commit, record, then switch back — or copy the
`.pt` files from a developer's working tree.
