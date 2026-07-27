"""Native-interface tests for the three DPA4 ablation switches (D4 / D1 / D2).

These are the tests that mlip-forge's ported component tests CANNOT cover: they are about how
the switches attach to OUR EquiformerV3, not about whether the ported math matches DPA4.

Handoff doc: ../mlip-forge/docs/handoff_eqv3_dpa4_d1d4_d2.md §6.2 asks for three of them.
This file implements those three plus a fourth that covers a hole found while verifying that
doc's claims (see `TestCheckpointTransferHole`).

Run:  ./.venv/bin/python -m pytest tests/experimental/equiformer_v3/test_dpa4_switches.py -q
"""
from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

from fairchem.experimental.models.equiformer_v3 import transformer_block as tb_mod
from fairchem.experimental.models.equiformer_v3.dpa4_ops import (
    EnvelopeGatedGraphSoftmax,
    check_dpa4_switch_transfer,
    dpa4_parameter_names,
    segment_envelope_gated_softmax,
)
from fairchem.experimental.models.equiformer_v3.envelope import (
    C3CutoffEnvelope,
    PolynomialEnvelope,
    build_envelope,
)
from fairchem.experimental.models.equiformer_v3.transformer_block import (
    EquivariantGraphAttention,
)

RCUT = 6.0


def _make_ga(**overrides):
    """A small GA matching the N2L2C64 recipe's attention shape (H=8, Cv=16, Ca=64)."""
    kwargs = dict(
        num_in_channels=64,
        num_hidden_channels=32,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        num_out_channels=64,
        lmax=2,
        mmax=2,
        so3_rotation=None,
        grid_resolution_list=[14, 8],
        max_num_elements=128,
        edge_channels_list=[10, 64, 64],
        activation="sep-merge_gates2_swiglu",
        softcap=None,
        attn_mask_rate=0.0,
    )
    kwargs.update(overrides)
    return EquivariantGraphAttention(**kwargs)


# =====================================================================================
# D4 -- envelope continuity. The whole reason D4 exists.
# =====================================================================================
class TestEnvelopeContinuity:
    @staticmethod
    def _derivs(env, r0, n=4):
        r = torch.tensor([r0], dtype=torch.float64, requires_grad=True)
        y = env(r).squeeze(-1).sum()
        out = []
        for _ in range(n):
            g = torch.autograd.grad(y, r, create_graph=True)[0]
            out.append(g.item())
            y = g.sum()
        return out

    def test_current_envelope_is_only_c2(self):
        """The defect D4 targets: our envelope's THIRD derivative jumps at the cutoff."""
        d = self._derivs(PolynomialEnvelope(cutoff=RCUT, exponent=5).double(), RCUT - 1e-5)
        assert abs(d[0]) < 1e-6, "value' should vanish at the cutoff"
        assert abs(d[1]) < 1e-4, "value'' should vanish at the cutoff"
        # -210 / rcut^3 -- the number the whole D4 argument rests on.
        assert d[2] * RCUT**3 == pytest.approx(-210.0, abs=1e-2)

    def test_c3_envelope_kills_the_third_order_jump(self):
        d = self._derivs(C3CutoffEnvelope(cutoff=RCUT, exponent=5).double(), RCUT - 1e-5)
        assert abs(d[0]) < 1e-8
        assert abs(d[1]) < 1e-6
        assert abs(d[2] * RCUT**3) < 1e-1, "third derivative must vanish (this IS D4)"

    @pytest.mark.parametrize("p", [3, 5, 7, 9])
    def test_c3_for_several_exponents(self, p):
        d = self._derivs(C3CutoffEnvelope(cutoff=RCUT, exponent=p).double(), RCUT - 1e-5)
        assert abs(d[2] * RCUT**3) < 1e-1

    def test_c3_closed_form_coefficients(self):
        e = C3CutoffEnvelope(cutoff=RCUT, exponent=5)
        assert (e.a, e.b, e.c, e.d) == (-56.0, 140.0, -120.0, 35.0)

    def test_both_envelopes_are_one_at_zero_and_zero_outside(self):
        r = torch.tensor([0.0, RCUT, RCUT + 1.0])
        for env in (PolynomialEnvelope(cutoff=RCUT), C3CutoffEnvelope(cutoff=RCUT)):
            v = env(r).squeeze(-1)
            assert v[0] == pytest.approx(1.0)
            assert v[1] == pytest.approx(0.0, abs=1e-7)
            assert v[2] == pytest.approx(0.0, abs=1e-7)

    def test_factory_default_is_the_unchanged_class(self):
        assert isinstance(build_envelope(cutoff=RCUT), PolynomialEnvelope)
        assert isinstance(build_envelope("dpa4_c3", cutoff=RCUT), C3CutoffEnvelope)
        with pytest.raises(ValueError):
            build_envelope("nope", cutoff=RCUT)


# =====================================================================================
# §6.2-1 -- defaults must be bit-identical
# =====================================================================================
class TestDefaultsAreBitIdentical:
    """If the defaults shift by one ULP, the existing baseline stops being a baseline."""

    def test_default_ga_uses_the_original_softmax_and_no_focus_compete(self):
        ga = _make_ga()
        assert type(ga.attn_softmax).__name__ == "GraphSoftmax"
        assert ga.focus_compete is None
        assert getattr(ga.attn_softmax, "folds_envelope", False) is False

    def test_default_ga_adds_no_parameters(self):
        base = dict(_make_ga().named_parameters())
        again = dict(_make_ga(attn_softmax_type="equiformerv3", focus_compete_groups=0).named_parameters())
        assert set(base) == set(again)
        assert dpa4_parameter_names(_make_ga()) == []

    def test_switching_on_adds_exactly_the_expected_keys(self):
        d1 = _make_ga(attn_softmax_type="dpa4_envelope_gated")
        new = set(dict(d1.named_parameters())) - set(dict(_make_ga().named_parameters()))
        assert new == {"attn_softmax.z_bias_raw"}
        assert d1.attn_softmax.z_bias_raw.numel() == 8   # one per head

        d2 = _make_ga(focus_compete_groups=2)
        new2 = set(dict(d2.named_parameters())) - set(dict(_make_ga().named_parameters()))
        assert new2 == {
            "focus_compete.adamw_focus_compete_w",
            "focus_compete.focus_compete_norm.adam_scale",
        }
        # H*Cv = 128 elements each, INDEPENDENT of F -> 256 per GA.
        assert sum(p.numel() for n, p in d2.named_parameters() if n in new2) == 256

    @pytest.mark.parametrize("groups", [2, 4, 8])
    def test_focus_param_count_is_independent_of_f(self, groups):
        ga = _make_ga(focus_compete_groups=groups)
        assert sum(p.numel() for n, p in ga.named_parameters() if "focus_compete" in n) == 256

    def test_focus_groups_must_divide_num_heads(self):
        with pytest.raises(ValueError, match="must divide num_heads"):
            _make_ga(focus_compete_groups=3)

    def test_unknown_softmax_type_raises(self):
        with pytest.raises(ValueError, match="unknown attn_softmax_type"):
            _make_ga(attn_softmax_type="whatever")


# =====================================================================================
# §6.2-2 -- the w-power contract. THE trap. See transformer_block.py's inline warning.
# =====================================================================================
class TestEnvelopePowerContract:
    """Swapping the softmax without the guard silently yields alpha ~ w^3 -- no symptom at all."""

    @staticmethod
    def _setup(n_edge=40, n_node=7, n_head=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(n_edge, n_head, generator=g)
        dst = torch.randint(0, n_node, (n_edge,), generator=g)
        w = torch.rand(n_edge, 1, generator=g)
        return logits, dst, w, n_node

    def test_baseline_softmax_does_not_fold_the_envelope(self):
        from fairchem.experimental.models.equiformer_v3.softmax import GraphSoftmax

        assert getattr(GraphSoftmax(), "folds_envelope", False) is False

    def test_dpa4_softmax_declares_that_it_folds(self):
        assert EnvelopeGatedGraphSoftmax(num_heads=8).folds_envelope is True

    def test_dpa4_adapter_matches_the_raw_operator(self):
        """The adapter must be a pure shape wrapper -- alpha ~ w^2, never w^3."""
        logits, dst, w, n_node = self._setup()
        ad = EnvelopeGatedGraphSoftmax(num_heads=8)
        got = ad(logits, index=dst, num_nodes=n_node, exp_rescale=w)
        want = segment_envelope_gated_softmax(
            logits=logits.reshape(-1, 1, 8),
            edge_env=w,
            dst=dst,
            n_nodes=n_node,
            z_bias_raw=ad.z_bias_raw,
            eps=ad.eps,
            src_weight=None,
        ).reshape(-1, 8)
        assert torch.equal(got, want)

    def test_guard_line_is_present_in_the_ga_source(self):
        """Literal source guard: nobody may delete the `folds_envelope` check unnoticed."""
        src = inspect.getsource(EquivariantGraphAttention.forward)
        assert "folds_envelope" in src, (
            "the folds_envelope guard vanished from GA.forward -- swapping in the DPA4 softmax "
            "would now produce alpha ~ w^3, an equivariant, normalised, loss-descending, "
            "completely meaningless model. Restore it."
        )
        assert "getattr(self.attn_softmax, 'folds_envelope'" in src

    @staticmethod
    def _halving_ratios(scale):
        """median alpha(w*scale/2) / alpha(w*scale) for both formulas, all neighbours receding."""
        from fairchem.experimental.models.equiformer_v3.softmax import GraphSoftmax

        logits, dst, w, n_node = TestEnvelopePowerContract._setup()
        w = w * scale
        base = GraphSoftmax()
        a1 = base(logits, index=dst, num_nodes=n_node, exp_rescale=w) * w
        a2 = base(logits, index=dst, num_nodes=n_node, exp_rescale=w * 0.5) * (w * 0.5)
        ad = EnvelopeGatedGraphSoftmax(num_heads=8)
        b1 = ad(logits, index=dst, num_nodes=n_node, exp_rescale=w)
        b2 = ad(logits, index=dst, num_nodes=n_node, exp_rescale=w * 0.5)
        return (a2 / a1).median().item(), (b2 / b1).median().item()

    def test_ours_decays_first_order_at_every_scale(self):
        """Ours is alpha ~ w EXACTLY, for any w: numerator w^2, denominator w^1, no floor."""
        for scale in (1.0, 0.1, 0.01, 1e-3):
            ours, _ = self._halving_ratios(scale)
            assert ours == pytest.approx(0.5, abs=0.01)

    def test_dpa4_reaches_second_order_only_asymptotically(self):
        """MEASURED, and it corrects the handoff doc's blanket "0.25" claim.

        DPA4's denominator is  zeta*exp(-max) + sum_j w_j^2 exp_j. Halving every w scales the
        SUM term by 0.25 but leaves zeta untouched, so the halving ratio is

            0.25 * (S + Z) / (0.25 S + Z)

        which is 0.25 only when Z >> S, i.e. deep in the tail near the cutoff. Measured here:

            w-scale  1.0   0.3   0.1    0.03   0.01
            ratio    0.59  0.30  0.256  0.251  0.250

        So DPA4 is second-order ONLY where w is small -- and at realistic w ~ O(1) it is in fact
        SHALLOWER than ours (0.59 vs 0.50). That is not a defect: smoothness is a property of the
        cutoff boundary, which is exactly where w is small. But any conclusion phrased as "DPA4
        attenuates attention faster" is wrong in the bulk, and must not be used to explain a
        result that comes from mid-range distances.
        """
        ours_far, dpa4_far = self._halving_ratios(1e-3)
        assert ours_far == pytest.approx(0.50, abs=0.01)
        assert dpa4_far == pytest.approx(0.25, abs=0.01), "asymptotic second-order decay"

        _, dpa4_bulk = self._halving_ratios(1.0)
        assert dpa4_bulk > 0.5, "at w ~ O(1) DPA4 is shallower than ours -- documented, not a bug"

    def test_attn_mask_rate_is_not_silently_dropped(self):
        with pytest.raises(ValueError, match="no counterpart"):
            _make_ga(attn_softmax_type="dpa4_envelope_gated", attn_mask_rate=0.1)


# =====================================================================================
# §6.2-3 -- one and only one envelope in a forward pass
# =====================================================================================
class TestSingleEnvelopeSite:
    def test_only_one_construction_site_in_the_training_path(self):
        """Two envelopes in one forward = two cutoffs mixed, silently, ablation dead.

        Scope is the model package only: `experimental/tasks/` holds standalone analysis
        scripts that legitimately build their own.
        """
        pkg = pathlib.Path(__file__).resolve().parents[3] / "experimental" / "models" / "equiformer_v3"
        hits = []
        for f in pkg.rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "build_envelope(" in stripped and "def build_envelope" not in stripped:
                    hits.append(f"{f.name}:{i}")
                if "PolynomialEnvelope(" in stripped or "C3CutoffEnvelope(" in stripped:
                    if f.name != "envelope.py":          # the registry / class defs live there
                        hits.append(f"{f.name}:{i}")
        assert len(hits) == 1, f"expected exactly one envelope construction site, found {hits}"
        assert hits[0].startswith("equiformer_v3.py")


# =====================================================================================
# EXTRA (not in the handoff doc) -- the checkpoint-transfer hole found while verifying it.
# =====================================================================================
class TestCheckpointTransferHole:
    """The handoff doc claims `strict=True` protects us. Verified: it does not.

    `load_pretrained_weights` builds its dict FROM the model, then overwrites only the keys the
    checkpoint happens to contain, then calls `load_state_dict` on that same dict. The key sets
    are equal by construction, so strict can never fire: a parameter present in the model but
    absent from the checkpoint is silently left at random init, with no log line.

    D1's z_bias_raw and D2's focus_compete.* are exactly such parameters, which is why the
    direct and grad-finetune stages must be switched identically -- and why we need our own check.
    """

    def test_loader_really_cannot_detect_missing_keys(self):
        src = inspect.getsource(
            __import__(
                "fairchem.experimental.trainers.equiformer_v3_dens_trainer",
                fromlist=["x"],
            )
        )
        assert "model_state_dict = self.model.state_dict()" in src, (
            "the loader changed -- re-verify whether missing DPA4 params are still silent"
        )

    def test_check_flags_a_stage_mismatch(self):
        ga_on = _make_ga(attn_softmax_type="dpa4_envelope_gated", focus_compete_groups=2)
        baseline_ckpt = dict(_make_ga().state_dict())          # stage-1 ran with switches OFF
        missing = check_dpa4_switch_transfer(ga_on, baseline_ckpt)
        assert set(missing) == {
            "attn_softmax.z_bias_raw",
            "focus_compete.adamw_focus_compete_w",
            "focus_compete.focus_compete_norm.adam_scale",
        }

    def test_check_passes_when_both_stages_match(self):
        ga_on = _make_ga(attn_softmax_type="dpa4_envelope_gated", focus_compete_groups=2)
        ckpt = {f"module.{k}": v for k, v in ga_on.state_dict().items()}   # DDP prefix survives
        assert check_dpa4_switch_transfer(ga_on, ckpt) == []


# =====================================================================================
# EXTRA -- HybridMuon routing (native-specific; DPA4 and mlip-forge never face this)
# =====================================================================================
class TestMuonRouting:
    def test_norm_gain_and_bias_are_kept_off_muon(self):
        """z_bias_raw and adam_scale are 2-D only incidentally; Newton-Schulz on them is nonsense."""
        from fairchem.core.common.muon import build_hybrid_muon_param_groups

        ga = _make_ga(attn_softmax_type="dpa4_envelope_gated", focus_compete_groups=2)
        no_wd = {"attn_softmax.z_bias_raw", "focus_compete.focus_compete_norm.adam_scale"}
        groups = build_hybrid_muon_param_groups(
            ga, no_weight_decay=no_wd, weight_decay=1e-3, adamw_lr=1e-4, muon_lr=1e-4
        )
        muon = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
        named = dict(ga.named_parameters())
        assert id(named["attn_softmax.z_bias_raw"]) not in muon
        assert id(named["focus_compete.focus_compete_norm.adam_scale"]) not in muon
        # the real linear map stays on Muon, as intended
        assert id(named["focus_compete.adamw_focus_compete_w"]) in muon
