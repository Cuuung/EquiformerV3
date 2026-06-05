"""HybridMuon optimizer: Muon for >=2D weight matrices, AdamW for everything else.

Muon (MomentUm Orthogonalized by Newton-schulz) replaces the raw momentum update of
a 2D weight matrix with its closest semi-orthogonal matrix, computed by a few steps of
a quintic Newton-Schulz iteration. It is only well-defined for matrices, so in practice
it is paired with AdamW for all the non-matrix parameters (biases, norm gains/scales,
and embeddings). That pairing is what we call "HybridMuon".

References
----------
* Keller Jordan et al., "Muon: An optimizer for the hidden layers of neural networks"
  (https://kellerjordan.github.io/posts/muon/) -- the Newton-Schulz coefficients and the
  ``max(1, rows/cols)**0.5`` RMS-matching scale below come from this work.

Notes for this codebase
-----------------------
* The Newton-Schulz iteration here orthogonalizes the **last two dimensions** and batches
  over any leading dimensions. That handles both ``nn.Linear`` weights ``(out, in)`` and
  EquiformerV3 ``SO3Linear`` weights ``(lmax+1, out, in)`` (one matrix per degree l).
* Under DDP each rank's gradients are all-reduced (averaged) before ``optimizer.step()``,
  so every rank computes the *same* Muon update and the replicas stay in sync. No
  distributed/ZeRO-aware Muon variant is required for the plain-DDP setup used here.
* This is a standard ``torch.optim.Optimizer`` subclass, so it is compatible with the
  trainer's AMP ``GradScaler`` path, ``clip_grad_norm_``, gradient accumulation, the
  ``LambdaLR`` scheduler, and checkpoint ``state_dict`` save/restore.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Return an approximately semi-orthogonal matrix with the same singular vectors as G.

    Orthogonalizes the last two dims of ``G`` (batched over any leading dims) via a
    quintic Newton-Schulz iteration run in bfloat16 for speed/stability. Works for any
    ``G.ndim >= 2``.
    """
    assert G.ndim >= 2, "Newton-Schulz requires a matrix (ndim >= 2)"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Operate on the wide orientation so X @ X^T is the smaller of the two products.
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    # Normalize so the spectral norm is <= 1 (per matrix in the batch).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class HybridMuon(torch.optim.Optimizer):
    """Hybrid Muon + AdamW optimizer.

    Each parameter group carries a boolean ``use_muon`` flag:
      * ``use_muon=True``  -> Muon update (Newton-Schulz orthogonalized momentum).
                              Uses: ``lr, momentum, nesterov, ns_steps, weight_decay``.
      * ``use_muon=False`` -> decoupled AdamW update.
                              Uses: ``lr, betas, eps, weight_decay``.

    Build the groups with :func:`build_hybrid_muon_param_groups` so the right parameters
    land in each group. Hyper-parameters not set on a group fall back to the constructor
    defaults below.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_muon=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            # Nesterov-style look-ahead: use grad + momentum * buf, else the raw buffer.
            g = grad.add(buf, alpha=momentum) if nesterov else buf
            update = zeropower_via_newtonschulz5(g, ns_steps)
            # RMS-match the update magnitude to AdamW-like scale for non-square matrices.
            scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("HybridMuon (AdamW path) does not support sparse grads")
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**t
            bias_correction2 = 1.0 - beta2**t
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def build_hybrid_muon_param_groups(
    model: torch.nn.Module,
    no_weight_decay: set[str],
    weight_decay: float,
    adamw_lr: float,
    muon_lr: float,
) -> list[dict]:
    """Split a model's parameters into Muon / AdamW groups.

    Rule (matches EquiformerV3's ``no_weight_decay`` semantics):
      * name in ``no_weight_decay``  -> AdamW, weight_decay=0  (biases, norms, embeddings)
      * else ``ndim >= 2``           -> Muon,  weight_decay=wd (Linear/SO3Linear weights)
      * else (leftover 1D)           -> AdamW, weight_decay=wd (rare)

    ``name.endswith(...)`` is used (as elsewhere in the trainer) so that DDP/compile
    prefixes like ``module.`` / ``_orig_mod.`` still match the unwrapped suffix names.
    """
    no_wd = set(no_weight_decay)
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(s) for s in no_wd):
            adamw_no_decay.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_decay.append(p)

    groups: list[dict] = []
    if muon_params:
        groups.append(
            dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay)
        )
    if adamw_decay:
        groups.append(
            dict(params=adamw_decay, use_muon=False, lr=adamw_lr, weight_decay=weight_decay)
        )
    if adamw_no_decay:
        groups.append(
            dict(params=adamw_no_decay, use_muon=False, lr=adamw_lr, weight_decay=0.0)
        )
    return groups
