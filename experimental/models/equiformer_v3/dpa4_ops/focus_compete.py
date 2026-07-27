# Ported (math kept verbatim) from the read-only reference
#   ../DPA4/deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/so2.py
#     - parameter definition : :1176-1206  (Step 7.5 "Optional cross-focus competition")
#     - forward math         : :1424-1438  (Step 6 "Cross-focus softmax competition")
#     - gate source          : :1334       (Step 4, `focus_gate_src = x_local[:, :, 0, :]`)
#     - defaults             : :794 focus_compete=True, :833 tau=1.0,
#                              :834 label_smoothing=0.02, :158 mlp_bias=False (-> NO bias)
#   commit 99c1ece2e5087c77267fba4ca84932b53621e42c
# via mlip-forge `_dpa4_ops/focus_compete.py`. The logic, scattered across DPA4's 965-line
# `SO2Convolution`, is extracted into a standalone nn.Module. No operator or constant was changed.
#
# ---- Mechanism --------------------------------------------------------------------------------
# The message channel axis is cut into F "focus" streams (C_wide = F * focus_dim). On EVERY EDGE,
# logits computed ONLY from the l=0 scalar channels do one softmax across the F streams; the
# resulting weight is broadcast to ALL (l, m) and ALL channels of that focus:
#
#     logits = einsum("efi,if->ef", ScalarRMSNorm(gate_src), W[Cf, F])   (+ bias)
#     alpha  = softmax(logits / tau, dim=focus)
#     alpha  = alpha * (1 - ls) + ls / F                                  # label smoothing
#     out    = x * alpha[..., None, None]
#
# EQUIVARIANCE: the logits come only from l=0 (rotation-invariant) scalars, so alpha is a per-edge
# scalar -- one and the same number multiplying every (l, m) component. It therefore commutes
# exactly with SO(3) rotations. This is deliberate in DPA4 (so2.py docstring :720-724).
#
# NOT a sparse MoE: every focus stream is computed in full, then softly weighted.
#
# ---- WARNING: amplitude semantics (must be carried into any conclusion) ------------------------
# softmax forces sum_F alpha = 1, so turning competition ON scales the total message amplitude
# down by roughly 1/F relative to "no gate". That is part of the mechanism (the F streams SHARE
# one budget and compete), not a bug -- DPA4 trains this way. Downstream norm + residual will
# absorb most of a constant scale, but NOT all of it. So a positive/negative D2 result cannot be
# attributed to "competition" alone without also considering the amplitude term. If D2 shows a
# clear effect, add an `alpha * F` (mean-normalised) control row to separate the two. Do NOT add
# that switch now -- one more knob is one more confound; run the faithful version first.
#
# ---- NATIVE-SPECIFIC NOTE (does not exist in DPA4 or mlip-forge) -------------------------------
# We train with HybridMuon (Muon on ndim>=2, AdamW on the rest). Both parameters here are 2-D and
# would default to the Muon group:
#   * `ScalarRMSNorm.adam_scale`  (F, Cf) is a NORM GAIN, not a linear map -- Newton-Schulz
#     orthogonalising it is meaningless. It is routed OUT of Muon via the model's
#     `no_weight_decay()` (see equiformer_v3.py), matching DPA4's `adam_` prefix (= Adam, no wd).
#   * `adamw_focus_compete_w` (Cf, F) IS a genuine linear map, so Muon is defensible and it is
#     LEFT on Muon. This ablation runs the moonshot (`moonlight`) scaling, whose update RMS is
#     0.2*lr independent of shape, so this small matrix does not get an oversized step (that
#     failure mode belongs to the keller/`ratio` scaling).
from __future__ import annotations

import torch
import torch.nn as nn

from .norm import ScalarRMSNorm


class CrossFocusCompetition(nn.Module):
    """DPA4/SeZM cross-focus softmax competition (Step 7.5 params + Step 6 forward).

    Args:
        n_focus: number of focus streams F (must be > 1; at F=1 the softmax is identically 1
            and DPA4 skips the branch outright).
        focus_dim: scalar width Cf of each focus stream (last dim of the gate source).
        tau: softmax temperature (DPA4 `focus_softmax_tau = 1.0`).
        label_smoothing: DPA4 `focus_label_smoothing = 0.02`; guarantees every stream keeps at
            least ls/F of the weight so none is starved early in training.
        eps: `ScalarRMSNorm` eps (DPA4 1e-7).
        bias: whether the logits carry a bias (DPA4 `mlp_bias`, default False).
        dtype: compute precision of the gate path (DPA4 uses compute_dtype = fp32).
    """

    def __init__(
        self,
        *,
        n_focus: int,
        focus_dim: int,
        tau: float = 1.0,
        label_smoothing: float = 0.02,
        eps: float = 1e-7,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_focus <= 1:
            raise ValueError(
                f"cross-focus competition needs n_focus > 1 (got {n_focus}); at F=1 the "
                "softmax is identically 1 and DPA4 skips the branch (so2.py:1424)."
            )
        self.n_focus = int(n_focus)
        self.focus_dim = int(focus_dim)
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)
        self.compute_dtype = dtype

        self.focus_compete_norm = ScalarRMSNorm(
            channels=self.focus_dim, n_focus=self.n_focus, eps=eps, dtype=dtype
        )
        # DPA4: nn.Parameter(empty(so2_focus_dim, n_focus)) + normal_(0, 0.01)
        self.adamw_focus_compete_w = nn.Parameter(
            torch.empty(self.focus_dim, self.n_focus, dtype=dtype)
        )
        nn.init.normal_(self.adamw_focus_compete_w, mean=0.0, std=0.01)
        # DPA4: only built when mlp_bias, zeros(n_focus)
        self.focus_compete_bias = (
            nn.Parameter(torch.zeros(self.n_focus, dtype=dtype)) if bias else None
        )

    def weights(self, gate_src: torch.Tensor) -> torch.Tensor:
        """Competition weights alpha, computed from the l=0 scalars only.

        Args:
            gate_src: (E, F, Cf) -- the message's l=0 scalar channels, sliced per focus.

        Returns:
            (E, F) non-negative weights; each row sums to 1 before label smoothing.
        """
        gate_src = gate_src.to(dtype=self.compute_dtype)
        focus_logits = torch.einsum(
            "efi,if->ef",
            self.focus_compete_norm(gate_src),
            self.adamw_focus_compete_w,
        )
        if self.focus_compete_bias is not None:
            focus_logits = focus_logits + self.focus_compete_bias.unsqueeze(0)
        alpha = torch.softmax(focus_logits / self.tau, dim=1)
        alpha = alpha * (1.0 - self.label_smoothing) + (
            self.label_smoothing / float(self.n_focus)
        )
        return alpha

    def forward(
        self, x: torch.Tensor, gate_src: torch.Tensor, focus_dim_index: int = 1
    ) -> torch.Tensor:
        """Apply cross-focus competition to x.

        Args:
            x: message containing a focus axis. DPA4's axis order is (E, F, D, Cf)
                (`focus_dim_index=1`); our GA's order is (E, D, F, Cf) (`focus_dim_index=2`).
                Only the broadcast position differs -- the math is identical.
            gate_src: (E, F, Cf) l=0 scalars.
            focus_dim_index: position of the focus axis inside x.

        Returns:
            Same shape as x.
        """
        alpha = self.weights(gate_src).to(dtype=x.dtype)  # (E, F)
        shape = [1] * x.dim()
        shape[0] = alpha.shape[0]
        shape[focus_dim_index] = self.n_focus
        return x * alpha.reshape(shape)

    def extra_repr(self) -> str:
        return (
            f"n_focus={self.n_focus}, focus_dim={self.focus_dim}, tau={self.tau}, "
            f"label_smoothing={self.label_smoothing}, "
            f"bias={self.focus_compete_bias is not None}"
        )
