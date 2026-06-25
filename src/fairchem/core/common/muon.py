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
  default ``max(1, rows/cols)**0.5`` RMS-matching scale (``update_scale="ratio"``).
* Jingyuan Liu et al. / Moonshot AI, "Muon is Scalable for LLM Training"
  (https://arxiv.org/abs/2502.16982) -- the ``0.2 * sqrt(max(rows, cols))`` shape-decoupled
  scale (``update_scale="moonlight"``) that makes a single muon_lr transferable across
  model sizes. Switch modes via ``optim.optimizer_params.update_scale``; re-calibrate
  muon_lr when you do (the two modes live on different magnitude scales).

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

import logging
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
        # --- Update-magnitude scaling mode. Selects how the orthogonalized update is
        #     rescaled before it is applied (see _muon_step for the math):
        #       "ratio"     : Keller-Jordan ``max(1, rows/cols)**0.5``. Update RMS ~
        #                     lr/sqrt(fan_in), so the EFFECTIVE per-matrix lr depends on
        #                     shape -> a safe muon_lr does NOT transfer across model sizes
        #                     (must be re-tuned whenever C / ffn_hidden / depth change).
        #       "moonlight" : Moonshot AI ``0.2 * sqrt(max(rows, cols))``. Update RMS ~
        #                     0.2*lr for EVERY matrix regardless of shape -> muon_lr is
        #                     ~transferable across widths/depths (calibrate it ONCE).
        #     NOTE: the two modes are on different magnitude scales, so switching modes
        #     REQUIRES re-calibrating muon_lr (a "ratio" muon_lr is meaningless here). ---
        update_scale: str = "ratio",
        # --- Muon-side divergence guard (acts where clip_grad_norm cannot: the
        #     orthogonalized update is norm-normalized, so a *global* grad clip cancels
        #     out for this group -- the only effective Muon protection is on the update
        #     itself). All knobs are config-overridable via optim.optimizer_params. ---
        skip_nonfinite: bool = True,
        spike_factor: float | None = 8.0,
        spike_ema_decay: float = 0.99,
        spike_warmup_steps: int = 200,
        # Max consecutive spike-skips a single matrix may take before we FORCE the step
        # through and re-baseline its EMA. This is the deadlock backstop: combined with the
        # "advance EMA even on a skip" fix in _muon_step, a matrix can never be frozen
        # forever (the old behavior froze EMA on skip -> once-skipped-always-skipped
        # ratchet that silently paralyzed ~75% of matrices on the 30M model). Set None to
        # rely on the EMA-advance recovery alone (slower but still non-deadlocking).
        spike_max_consecutive_skips: int | None = 25,
        # --- Absolute runaway backstop. The EMA-relative spike guard above is RELATIVE and
        #     SELF-DISARMS during a SLOW runaway: as the gradient climbs over many steps the
        #     per-matrix EMA climbs with it (skips advance the EMA -- the deadlock fix), so
        #     the 8x-EMA threshold keeps rising and the diverging steps pass as "no skip".
        #     This is exactly how the 06-20 mlr=2e-4 run silently lobotomized: max NS-input
        #     RMS ramped 0.12 -> 9.4 over ~0.5ep, never produced a NaN (so no trainer crash),
        #     and ran to ep35 dead (val flat 9.96 for 11 epochs). This backstop is ABSOLUTE
        #     (does NOT drift with the EMA): if the per-step MAX NS-input RMS stays above
        #     `spike_abs_threshold` for `spike_abs_max_consecutive` consecutive optimizer
        #     steps, ABORT the run (raise). best_checkpoint holds the pre-runaway peak, so
        #     aborting loses nothing and fails fast instead of wasting epochs on a dead model.
        #     Healthy training kept max RMS <=~2.4 momentarily, so 4.0 SUSTAINED is
        #     unambiguously a runaway (a 1-step transient won't trip it). Single transient
        #     matrix spikes are handled by the relative guard; this targets only a sustained
        #     network-wide ramp. Set spike_abs_threshold=None to disable. ---
        spike_abs_threshold: float | None = 4.0,
        spike_abs_max_consecutive: int = 10,
        log_every: int = 200,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if update_scale not in ("ratio", "moonlight"):
            raise ValueError(
                f"Invalid update_scale: {update_scale!r} (expected 'ratio' or 'moonlight')"
            )
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

        # Update-magnitude scaling mode (global, not per-group).
        self.update_scale = update_scale
        # Divergence-guard config (global, not per-group).
        self.skip_nonfinite = skip_nonfinite
        self.spike_factor = spike_factor
        self.spike_ema_decay = spike_ema_decay
        self.spike_warmup_steps = spike_warmup_steps
        self.spike_max_consecutive_skips = spike_max_consecutive_skips
        self.spike_abs_threshold = spike_abs_threshold
        self.spike_abs_max_consecutive = spike_abs_max_consecutive
        self.log_every = log_every
        # Per-step Muon stats (reset each step()).
        self._global_muon_steps = 0
        self._muon_skipped = 0
        self._muon_nonfinite = 0
        self._muon_spike = 0
        self._muon_grad_rms_max = 0.0
        # Consecutive optimizer steps whose max NS-input RMS exceeded spike_abs_threshold
        # (absolute runaway backstop; reset to 0 on any below-threshold step).
        self._abs_streak = 0
        # Log only on the master rank (every rank computes identical updates under DDP).
        try:
            from fairchem.core.common import distutils as _du

            self._is_master = _du.is_master()
        except Exception:
            self._is_master = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reset per-step Muon stats.
        self._muon_skipped = 0
        self._muon_nonfinite = 0
        self._muon_spike = 0
        self._muon_grad_rms_max = 0.0
        ran_muon = False
        for group in self.param_groups:
            if group["use_muon"]:
                ran_muon = True
                self._muon_step(group)
            else:
                self._adamw_step(group)
        if ran_muon:
            self._global_muon_steps += 1
            if self._is_master:
                # Always surface skips immediately (early divergence warning); otherwise
                # emit a periodic RMS summary every `log_every` steps.
                if self._muon_skipped:
                    logging.info(
                        f"[HybridMuon] step {self._global_muon_steps}: SKIPPED "
                        f"{self._muon_skipped} matrices "
                        f"(nonfinite={self._muon_nonfinite}, spike={self._muon_spike}); "
                        f"max NS-input RMS={self._muon_grad_rms_max:.3e}"
                    )
                elif self.log_every and self._global_muon_steps % self.log_every == 0:
                    logging.info(
                        f"[HybridMuon] step {self._global_muon_steps}: "
                        f"max NS-input RMS={self._muon_grad_rms_max:.3e} (no skips)"
                    )
            # --- Absolute runaway backstop (see __init__): catch a SUSTAINED network-wide
            #     ramp of the NS-input RMS that the EMA-relative guard cannot (its threshold
            #     drifts up with the runaway). Counts CONSECUTIVE steps over the absolute
            #     ceiling and ABORTS past the cap. Gated behind spike_warmup_steps so early
            #     high-grad warmup steps never trip it. Runs on every rank (identical state
            #     under DDP), so the whole job aborts together. ---
            if (
                self.spike_abs_threshold is not None
                and self._global_muon_steps >= self.spike_warmup_steps
            ):
                if self._muon_grad_rms_max > self.spike_abs_threshold:
                    self._abs_streak += 1
                    if self._is_master:
                        logging.warning(
                            f"[HybridMuon] ABSOLUTE spike: max NS-input RMS="
                            f"{self._muon_grad_rms_max:.3e} > {self.spike_abs_threshold} "
                            f"(streak {self._abs_streak}/{self.spike_abs_max_consecutive}, "
                            f"step {self._global_muon_steps})"
                        )
                    if self._abs_streak >= self.spike_abs_max_consecutive:
                        raise RuntimeError(
                            f"[HybridMuon] ABORT: max NS-input RMS stayed above "
                            f"{self.spike_abs_threshold} for {self._abs_streak} consecutive "
                            f"steps (now {self._muon_grad_rms_max:.3e}, step "
                            f"{self._global_muon_steps}) -- a sustained runaway the "
                            f"EMA-relative guard cannot stop. best_checkpoint holds the "
                            f"pre-runaway peak: lower muon_lr and resume from it."
                        )
                else:
                    self._abs_streak = 0
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

            # --- Guard 1: non-finite gradient. Skip WITHOUT touching the momentum buffer,
            #     so a transient NaN/Inf cannot permanently poison `buf` (which would turn
            #     every subsequent step into NaN -- the fp16 "death spiral"). ---
            if self.skip_nonfinite and not torch.isfinite(grad).all():
                self._muon_skipped += 1
                self._muon_nonfinite += 1
                continue

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            # Candidate momentum update -- computed OUT-OF-PLACE so that if we decide to
            # skip below, `buf` (and hence future steps) stays uncontaminated by the spike.
            cand = buf.mul(momentum).add(grad)
            # Nesterov-style look-ahead: use grad + momentum * cand, else the raw buffer.
            g = grad.add(cand, alpha=momentum) if nesterov else cand

            # RMS of the Newton-Schulz input -- the quantity that actually carries the
            # spike signal (the post-orthogonalization update is ~unit-norm by design).
            g_rms = float(g.pow(2).mean().sqrt())
            if g_rms > self._muon_grad_rms_max:
                self._muon_grad_rms_max = g_rms
            ema = state.get("g_rms_ema")

            # --- Guard 2: EMA-relative spike. If this matrix's NS-input RMS jumps far above
            #     its own running average, skip the step (buffer + param left unchanged) so a
            #     transient does not corrupt the weights. Self-calibrating -> no absolute
            #     threshold to hand-tune.
            #
            #     RECOVERY (deadlock fix): the previous version froze the EMA on a skip, so a
            #     *sustained* regime shift (e.g. the post-8ep loss bump) kept g_rms above the
            #     stale low EMA forever -> the matrix was skipped every step and could never
            #     recover. ~75% of the 30M model's matrices fell into this ratchet, silently
            #     paralyzing training (no NaN, loss not diverging, forces just stopped
            #     learning). Two changes break it:
            #       1. ADVANCE the EMA even on a skip (with the finite g_rms): a one-off spike
            #          barely moves it (decay 0.99) so genuine spikes are still rejected next
            #          step, but a sustained shift pulls the baseline up so the threshold
            #          recovers within a handful of steps.
            #       2. After spike_max_consecutive_skips skips on one matrix, FORCE the step
            #          through and re-baseline its EMA -> a hard cap on how long any matrix
            #          can be frozen. ---
            spiking = (
                self.spike_factor is not None
                and ema is not None
                and self._global_muon_steps >= self.spike_warmup_steps
                and g_rms > self.spike_factor * ema
            )
            if spiking:
                streak = state.get("spike_streak", 0) + 1
                cap = self.spike_max_consecutive_skips
                if cap is None or streak <= cap:
                    state["spike_streak"] = streak
                    # Advance the EMA on the skip (recovery fix #1).
                    state["g_rms_ema"] = (
                        self.spike_ema_decay * ema
                        + (1.0 - self.spike_ema_decay) * g_rms
                    )
                    self._muon_skipped += 1
                    self._muon_spike += 1
                    if self._is_master:
                        logging.warning(
                            f"[HybridMuon] spike SKIP matrix {tuple(p.shape)}: "
                            f"g_rms={g_rms:.3e} > {self.spike_factor}x EMA={ema:.3e} "
                            f"(streak {streak}, muon step {self._global_muon_steps})"
                        )
                    continue
                # Streak hit the cap: treat the elevated level as the new regime, not a
                # transient -> force-commit and re-baseline so we can never deadlock (fix #2).
                if self._is_master:
                    logging.warning(
                        f"[HybridMuon] spike RECOVER matrix {tuple(p.shape)} after {cap} "
                        f"consecutive skips: force-commit, re-baseline EMA -> g_rms={g_rms:.3e} "
                        f"(muon step {self._global_muon_steps})"
                    )
                ema = g_rms  # re-baseline; the committed EMA-update below collapses to g_rms

            # Committed (or force-recovered) step -> clear the skip streak.
            state["spike_streak"] = 0

            # Commit momentum, then apply the orthogonalized update.
            buf.copy_(cand)
            update = zeropower_via_newtonschulz5(g, ns_steps)
            # RMS-match the update magnitude. The NS output is ~semi-orthogonal, so its
            # per-element RMS ~ 1/sqrt(max(rows, cols)); the scale below decides how that
            # turns into the actual step (see __init__ `update_scale` for the trade-off).
            if self.update_scale == "moonlight":
                # 0.2*sqrt(max(rows,cols)) * (1/sqrt(max(rows,cols))) = 0.2 -> update RMS
                # ~0.2*lr for every matrix, independent of shape (muon_lr transferable).
                scale = 0.2 * math.sqrt(max(g.size(-2), g.size(-1)))
            else:  # "ratio": Keller-Jordan; update RMS ~ lr/sqrt(fan_in) (shape-dependent).
                scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(update, alpha=-lr * scale)

            # Update the per-matrix EMA of the NS-input RMS for this committed step. (Skipped
            # steps advance the EMA separately above; force-recovered steps set ema=g_rms so
            # this line collapses to a clean re-baseline.)
            state["g_rms_ema"] = (
                g_rms
                if ema is None
                else self.spike_ema_decay * ema + (1.0 - self.spike_ema_decay) * g_rms
            )

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
