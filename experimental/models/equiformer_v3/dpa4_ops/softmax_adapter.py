"""Adapter making DPA4's envelope-gated softmax a drop-in replacement for our `GraphSoftmax`,
so that `EquivariantGraphAttention` can run a SINGLE-FACTOR attention-normalisation ablation.

Underlying operator: `.attention.segment_envelope_gated_softmax`
(ported verbatim from DPA4/SeZM `sezm_nn/attention.py`, commit 99c1ece2).

## Why an adapter and not a straight swap

The two sides draw the responsibility boundary in DIFFERENT places:

- ours: `GraphSoftmax` computes `w*exp(l) / sum w*exp(l)` -- the SECOND `w` in the numerator is
  multiplied on OUTSIDE, by the GA (`transformer_block.py`, right after the softmax call).
- DPA4: `segment_envelope_gated_softmax` folds `w^2` into numerator AND denominator itself and
  returns the final alpha.

So this adapter does two things:
 1. fixes up shapes (our logits are `[E, H]`, the DPA4 operator wants `[E, F, H]`; we use F=1);
 2. declares `folds_envelope = True` so the GA knows NOT to multiply by `w` a second time.

## !!! The trap this class exists to prevent !!!

Swapping the softmax without honouring `folds_envelope` yields `alpha ~ w^3` -- neither
EquiformerV3 nor DPA4, a third thing that is nobody's formula. And it has NO visible symptom:
still equivariant, still normalised, loss still goes down, training still finishes, kappa still
comes out. The resulting ablation row would be pure noise and nobody would know.
`tests/models/equiformer_v3/test_dpa4_switches.py` pins the guard (both directions, plus a
literal source guard so the `getattr` check cannot be deleted unnoticed).

## Single-factor discipline

Turning this on changes ONLY the attention-weight normalisation formula. Logit construction
(alpha_norm / alpha_act / alpha_dot), the value projection, the aggregation, and the envelope
itself are untouched. `softcap` is still applied BEFORE normalisation, same order as the
EquiformerV3 path, so softcap-presence is not folded into this ablation.

`attn_mask_rate` (our `GraphSoftmax.exp_dropout`) has NO counterpart in the DPA4 formula --
forcing it in would break the "numerator and denominator carry the same weight" structure. Our
recipes use 0; anything else raises rather than silently degrading.
"""
from __future__ import annotations

import torch

from torch_geometric.utils.num_nodes import maybe_num_nodes

from ..softmax import SoftCap
from .attention import segment_envelope_gated_softmax

# softplus(0.5413) ~= 1.0 -- DPA4's init (so2.py:1138-1147), "initial competitive equilibrium".
_Z_BIAS_RAW_INIT = 0.5413


class EnvelopeGatedGraphSoftmax(torch.nn.Module):
    """DPA4 destination-wise envelope-gated softmax; signature matches `GraphSoftmax.forward`.

    The returned alpha ALREADY contains the `w^2` factor -- callers must not multiply by the
    envelope again. That contract is advertised by the `folds_envelope` class attribute.

    Args:
        num_heads: number of attention heads H (sets the shape of `z_bias_raw`, `[1, H]`).
        eps: denominator stabiliser (same name/position as our `GraphSoftmax`, default 1e-16).
        softcap: as in the EquiformerV3 path, a tanh soft-clamp applied to the logits BEFORE
            normalisation; None disables it.
        exp_dropout: must be 0 (no DPA4 counterpart; raises otherwise).
    """

    #: Tells callers: the envelope is already folded into the numerator, do not multiply again.
    folds_envelope: bool = True

    def __init__(self, num_heads: int, eps: float = 1e-16, softcap=None,
                 exp_dropout: float = 0.0) -> None:
        super().__init__()
        if exp_dropout:
            raise ValueError(
                "DPA4 envelope-gated softmax has no counterpart for exp_dropout "
                f"(attn_mask_rate); got {exp_dropout}. Our recipes use 0 -- refusing to "
                "silently degrade."
            )
        self.num_heads = num_heads
        self.eps = float(eps)
        self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()
        # Shape (F, H) with F=1: we do not use DPA4's multi-focus axis here (that is D2, a
        # separate independent switch).
        self.z_bias_raw = torch.nn.Parameter(
            torch.full((1, num_heads), _Z_BIAS_RAW_INIT)
        )

    def forward(
        self,
        src,
        index=None,
        ptr=None,
        num_nodes=None,
        dim=0,
        exp_rescale=None,
    ):
        if ptr is not None:
            raise NotImplementedError("ptr/segment path not implemented (attention never uses it).")
        if index is None:
            raise NotImplementedError("index (destination node ids) is required.")
        if dim != 0:
            raise NotImplementedError(f"only dim=0 is supported, got {dim}.")

        src = self.softcap(src)
        n_edge, n_head = src.shape
        if n_head != self.num_heads:
            raise ValueError(f"num_heads mismatch: built with {self.num_heads}, got {n_head}.")
        # ============== NEVER int() A SHAPE HERE -- IT SILENTLY BREAKS COMPILE ================
        # The conservative-force path make_fx-traces the model with a DUMMY prime-shaped example
        # (compile_utils.make_prime_graph_example: natoms=7, nedges=11). `int()` on a symbolic
        # size materialises that trace-time value, so `torch.zeros(n_nodes, ...)` below would bake
        # a literal 7 into the graph while every other node dim stays symbolic -- and the run dies
        # with "size of tensor a (7) must match tensor b (s77: hint = 431)". That is exactly how
        # the first D1 grad-ft submission failed. `maybe_num_nodes` is what the baseline
        # GraphSoftmax uses (softmax.py:65) and passes the size through UNTOUCHED, keeping it
        # symbolic. test_dpa4_symbolic_trace.py pins this by tracing at two different node counts.
        n_nodes = maybe_num_nodes(index, num_nodes)

        if exp_rescale is None:
            # use_envelope=False: degenerates to "softmax with a learnable leak zeta" (w == 1),
            # which is a legitimate limit of the DPA4 formula, not a silent change of semantics.
            edge_env = src.new_ones(n_edge, 1)
        else:
            edge_env = exp_rescale

        alpha = segment_envelope_gated_softmax(
            logits=src.reshape(n_edge, 1, n_head),
            edge_env=edge_env,
            dst=index,
            n_nodes=n_nodes,
            z_bias_raw=self.z_bias_raw,
            eps=self.eps,
            src_weight=None,  # zone-bridging SFPG; we have none
        )
        return alpha.reshape(n_edge, n_head)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, eps={self.eps}, folds_envelope=True"
