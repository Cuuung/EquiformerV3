# Ported VERBATIM (zero math changes) from the read-only reference
#   ../DPA4/deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/attention.py
#   commit 99c1ece2e5087c77267fba4ca84932b53621e42c  (DPA4 = deepmd-kit fork, internal name SeZM)
# via mlip-forge `_dpa4_ops/attention.py`. The source file is a pure function with ZERO deepmd
# imports (only torch / F), so nothing had to be stripped. The original
# `@torch.amp.autocast("cuda", enabled=False)` is kept (forces fp32 for the softmax; a no-op on CPU).
#
# ---- What this changes relative to our GraphSoftmax (the whole point) -----------------------
# With w = edge envelope, l = logits, destination-wise normalisation:
#
#   ours (equiformer_v3/softmax.py + transformer_block.py:323-324):
#       alpha = w^2 * exp(l) / (sum_j w_j * exp(l_j) + eps)
#       -- GraphSoftmax(exp_rescale=w) folds ONE w into numerator AND denominator; the GA then
#          multiplies by w a SECOND time outside. So the denominator carries only w^1.
#   DPA4 (this file):
#       alpha = w^2 * exp(l) / (zeta * exp(-max) + sum_j w_j^2 * exp(l_j)),  zeta = softplus(z_bias) > 0
#
# Two substantive differences: (1) the denominator also carries w^2; (2) zeta is a LEARNABLE
# strictly-positive denominator floor that does not collapse when the neighbours recede.
#
# Consequence, verified by hand on our formula: when ALL neighbours recede together, ours decays
# as alpha ~ w (denominator collapses in step) while DPA4 decays as alpha ~ w^2/zeta.
# When a SINGLE edge leaves the cutoff with the others fixed, both give alpha_e ~ w_e^2 -- but
# our DENOMINATOR sum_j w_j exp_j then has a non-vanishing third derivative at the cutoff
# (proportional to the envelope's -210/rcut^3), which every OTHER neighbour's alpha inherits.
# DPA4's w^2 ~ (1-x)^6 kills that third-order kink. Since FC3 (and therefore kappa_SRME) IS a
# third-derivative object, this is the mechanism worth testing -- it is the same third-order
# defect that D4 (C3 envelope) attacks from the other side, which is why the two are packaged.
#
# ---- Not ported ------------------------------------------------------------------------------
# `src_weight` is DPA4's zone-bridging SFPG gate (removes a source from numerator AND denominator
# simultaneously). We have no zone-bridging and always pass None. The parameter is kept so the
# function stays verbatim-identical to DPA4 for parity testing.
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def segment_envelope_gated_softmax(
    logits: torch.Tensor,
    edge_env: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    z_bias_raw: torch.Tensor,
    eps: float,
    src_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute destination-wise envelope-gated softmax attention.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H).
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes.
    z_bias_raw
        Unconstrained denominator bias with shape (F, H).
        Softplus is applied to keep the bias strictly positive.
    eps
        Small epsilon for denominator stability.
    src_weight
        Optional per-edge source-side multiplier with shape (E, 1) or
        (E,). When provided the per-edge weight becomes
        ``edge_env**2 * src_weight``.

    Returns
    -------
    torch.Tensor
        Normalized edge weights with shape (E, F, H).
    """
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    eps_f = float(eps)

    # === Step 1. Flatten (F, H) and build the effective per-edge weight ===
    logits_2d = logits.reshape(n_edge, n_channel)
    edge_env_1d = edge_env.squeeze(-1).to(dtype=logits.dtype).clamp_min(0.0)
    # edge_weight_sq is the non-negative factor multiplying every exp(logit). Folding
    # src_weight in here guarantees a src_weight=0 edge is removed from group max /
    # numerator / denominator in ONE place.
    edge_weight_sq = edge_env_1d.square()
    if src_weight is not None:
        edge_weight_sq = (
            edge_weight_sq
            * src_weight.reshape(n_edge).to(dtype=logits.dtype).clamp_min(0.0)
        )
    zeta = F.softplus(z_bias_raw).reshape(1, n_channel).to(dtype=logits.dtype)
    dst_index = dst.reshape(n_edge, 1).expand(n_edge, n_channel)
    has_weight = edge_weight_sq > 0.0
    logits_for_max = torch.where(
        has_weight.reshape(n_edge, 1),
        logits_2d,
        torch.full_like(logits_2d, float("-inf")),
    )

    # === Step 2. Destination-wise max for stable exponentials ===
    group_max = torch.full(
        (n_nodes, n_channel),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    group_max = torch.scatter_reduce(
        group_max,
        0,
        dst_index,
        logits_for_max,
        reduce="amax",
        include_self=True,
    )
    edge_max = group_max.index_select(0, dst)
    edge_max = torch.where(
        torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max)
    )
    group_max_safe = torch.where(
        torch.isfinite(group_max), group_max, torch.zeros_like(group_max)
    )

    # === Step 3. Envelope/SFPG-gated exponential terms ===
    exp_shifted = torch.exp(logits_2d - edge_max)
    edge_weighted_exp = edge_weight_sq.reshape(n_edge, 1) * exp_shifted

    # === Step 4. Destination-wise normalization with positive denominator bias ===
    denom_sum = torch.zeros(
        n_nodes,
        n_channel,
        dtype=logits.dtype,
        device=logits.device,
    )
    denom_sum = torch.scatter_add(denom_sum, 0, dst_index, edge_weighted_exp)
    denom = denom_sum + zeta * torch.exp(-group_max_safe)

    alpha = edge_weighted_exp / (denom.index_select(0, dst) + eps_f)
    return alpha.reshape(n_edge, n_focus, n_head)
