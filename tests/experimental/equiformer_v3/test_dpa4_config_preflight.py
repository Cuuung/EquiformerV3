"""PREFLIGHT: catch config errors locally instead of on the pool.

Written after the first A0/A1 submission died at startup: both inherited
`optim.use_compile: True` (the OUTER old torch.compile) from the N2L2C64 baseline, while the
new stack sets `model.enable_compile: True`. The trainer hard-fails on that pair
(equiformer_v3_dens_trainer.py:394) -- a 10-second failure that cost a pool round-trip because
nothing checked it locally.

This file replicates every startup guard the trainer applies, plus the ones the model
constructor applies, and runs them over the ablation configs. It does NOT need a GPU, a
dataset, or torch>=2.11 (the compile flags are validated as CONFIG, then stripped before
instantiating, since compile itself needs the pool image).

Run:  PYTHONPATH=src ./.venv/bin/python -m pytest tests/experimental/equiformer_v3/test_dpa4_config_preflight.py -q
"""
from __future__ import annotations

import pathlib

import pytest
import torch
import yaml

from fairchem.core.common.registry import registry
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401  (register)
import fairchem.experimental.models.equiformer_v3.equiformer_v3_dens  # noqa: F401
from fairchem.experimental.models.equiformer_v3.dpa4_ops import dpa4_parameter_names

REPO = pathlib.Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO / "experimental" / "configs" / "omat24" / "mptrj" / "experiments"


def _ablation_configs():
    """The A0/A1/A2 arm configs ONLY (the '..._bf16-compile.yml' matched-stack set).

    The dpa4_ablation dir also holds deliberately-different configs that are NOT part of the
    matched A/B and must be excluded from these preflight assertions:
      * *_GRADFP32-eager.yml  -- the E-G probe (grad stage on purpose reverted to fp32/no-compile)
      * keller_*_KAPPA-AB.yml  -- the keller vs moonshot gradft (fp32, no compile, different base)
    """
    return sorted(
        p for p in CONFIG_ROOT.glob("*/dpa4_ablation/dpa4_A*_bf16-compile.yml")
    )


def _load(p):
    return yaml.safe_load(p.read_text())


def test_configs_exist():
    assert len(_ablation_configs()) == 6, "expected 3 arms x 2 stages"


@pytest.mark.parametrize("path", _ablation_configs(), ids=lambda p: p.name[:48])
class TestTrainerStartupGuards:
    """Every `raise` the trainer can hit before the first step."""

    def test_outer_and_inner_compile_are_not_both_on(self, path):
        """equiformer_v3_dens_trainer.py:394 -- THE bug that killed the first submission."""
        c = _load(path)
        assert not (
            c["optim"].get("use_compile", False) and c["model"].get("enable_compile", False)
        ), (
            "optim.use_compile (outer torch.compile) and model.enable_compile (in-model make_fx) "
            "are mutually exclusive; the trainer raises at startup. maoruicong's stack replaces "
            "the outer compile with the in-model one, so use_compile must be False."
        )

    def test_fp16_amp_and_bf16_amp_are_not_both_on(self, path):
        """equiformer_v3_dens_trainer.py:~400 -- optim.amp (fp16+GradScaler) vs model.use_amp (bf16)."""
        c = _load(path)
        assert not (c["optim"].get("amp", False) and c["model"].get("use_amp", False))

    def test_matmul_precision_is_valid(self, path):
        c = _load(path)
        mp = c["optim"].get("matmul_precision", "highest")
        assert mp in ("highest", "high", "medium")

    def test_gradient_checkpointing_list_matches_depth(self, path):
        c = _load(path)
        gc = c["model"].get("gradient_checkpointing_block_list")
        if gc is not None:
            assert len(gc) == c["model"]["num_layers"]


@pytest.mark.parametrize("path", _ablation_configs(), ids=lambda p: p.name[:48])
def test_model_actually_constructs(path):
    """Constructor-level guards: focus divisibility, attn_mask_rate, unknown switch strings."""
    c = _load(path)
    m = dict(c["model"])
    name = m.pop("name")
    # compile needs torch>=2.11 (pool image only); validated as config above, stripped here.
    m.pop("enable_compile", None)
    m.pop("compile_dynamic", None)
    torch.manual_seed(0)
    model = registry.get_model_class(name)(**m)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_the_stack_is_actually_enabled_everywhere():
    """The whole point of these arms is that they carry maoruicong's stack."""
    for p in _ablation_configs():
        c = _load(p)
        assert c["model"].get("enable_compile") is True, p.name
        assert c["model"].get("compile_dynamic") is True, p.name
        assert c["optim"].get("matmul_precision") == "high", p.name
        # bf16 blocks on direct, fp32 blocks on grad-ft (matches the production 30M recipe)
        expected_amp = "direct" in str(p)
        assert c["model"].get("use_amp") is expected_amp, p.name


def test_arms_differ_ONLY_in_the_dpa4_switches():
    """A/B discipline: if anything else drifted between arms, the comparison is void."""
    SWITCHES = {"envelope_type", "attn_softmax_type", "focus_compete_groups"}
    for stage in ("direct", "gradient"):
        cfgs = {
            p.name.split("_")[1]: _load(p)
            for p in CONFIG_ROOT.glob(f"{stage}/dpa4_ablation/dpa4_A*_bf16-compile.yml")
        }
        base = cfgs["A0-baseline"]
        for arm, c in cfgs.items():
            if arm == "A0-baseline":
                continue
            for section in ("model", "optim", "loss_functions", "dataset", "outputs"):
                a, b = base.get(section), c.get(section)
                if isinstance(a, dict):
                    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                    assert diff <= SWITCHES, f"{stage}/{arm}: unexpected drift in {section}: {diff}"
                else:
                    assert a == b, f"{stage}/{arm}: {section} differs"


def test_switches_are_identical_across_the_two_stages():
    """Mandatory: the checkpoint loader CANNOT detect a mismatch (strict=True is vacuous)."""
    SWITCHES = ("envelope_type", "attn_softmax_type", "focus_compete_groups")
    for arm in ("A0-baseline", "A1-D1D4", "A2-D2F2"):
        d = _load(next(CONFIG_ROOT.glob(f"direct/dpa4_ablation/dpa4_{arm}_*_bf16-compile.yml")))
        g = _load(next(CONFIG_ROOT.glob(f"gradient/dpa4_ablation/dpa4_{arm}_*_bf16-compile.yml")))
        for s in SWITCHES:
            assert d["model"].get(s) == g["model"].get(s), (
                f"{arm}: switch {s} differs between stages -- stage-2 would silently train "
                f"half-random parameters (see dpa4_ops/__init__.py)"
            )


def test_expected_new_parameter_counts_per_arm():
    """N2L2C64 = 2 blocks, num_heads=8, attn_value_channels=16."""
    expected = {"A0-baseline": 0, "A1-D1D4": 16, "A2-D2F2": 512}
    for arm, want in expected.items():
        for stage in ("direct", "gradient"):
            p = next(CONFIG_ROOT.glob(f"{stage}/dpa4_ablation/dpa4_{arm}_*_bf16-compile.yml"))
            m = dict(_load(p)["model"])
            name = m.pop("name")
            m.pop("enable_compile", None)
            m.pop("compile_dynamic", None)
            torch.manual_seed(0)
            model = registry.get_model_class(name)(**m)
            got = sum(
                p_.numel()
                for n_, p_ in model.named_parameters()
                if n_ in set(dpa4_parameter_names(model))
            )
            assert got == want, f"{arm}/{stage}: expected +{want} DPA4 params, got +{got}"
