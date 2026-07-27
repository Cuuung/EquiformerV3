# Ported (math kept verbatim) from the read-only reference
#   ../DPA4/deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/norm.py:545 `ScalarRMSNorm`
#   commit 99c1ece2e5087c77267fba4ca84932b53621e42c
# via mlip-forge `_dpa4_ops/norm.py`. Only the deepmd dependencies were stripped:
#   - `deepmd.pt.utils.env.DEVICE` -> left to PyTorch's default device (params follow module .to())
#   - `compute_dtype` (deepmd PRECISION_DICT) -> plain `dtype` kwarg, default fp32, matching what
#     DPA4 actually passes (the gate path is forced to fp32 for numerical stability)
#   - `serialize` / `deserialize` (deepmd checkpoint protocol) -> not ported; we use fairchem's.
#
# The `@torch.amp.autocast("cuda", enabled=False)` is kept, exactly as in DPA4: under AMP the RMS
# is still computed in fp32 (gate logits are precision-sensitive). No-op on CPU.
# In DPA4 `eps` is a `persistent=False` buffer -> not in state_dict; here it is a plain float, so
# the state_dict key set matches DPA4 (only `adam_scale`).
from __future__ import annotations

import torch
import torch.nn as nn


class ScalarRMSNorm(nn.Module):
    """SeZM's per-focus scalar RMSNorm (no bias).

    `n_focus=1` degenerates to a single stream; with `n_focus>1` each focus stream owns an
    independent learnable scale. The DPA4 comment is explicit: "Bias is intentionally omitted
    to keep the gate paths minimal."

    Args:
        channels: feature width C of the last dimension.
        n_focus: number of focus streams F.
        eps: numerical stabiliser (DPA4 default 1e-7).
        dtype: compute precision (DPA4 passes compute_dtype = fp32).
    """

    def __init__(
        self,
        *,
        channels: int,
        n_focus: int = 1,
        eps: float = 1e-7,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.compute_dtype = dtype
        # DPA4's `adam_` prefix is its own optimizer-routing marker (Adam, no weight decay).
        # We keep the name only so the state_dict keys line up with DPA4 for parity checks.
        # NOTE for our HybridMuon: this is a 2-D tensor, so it WOULD be routed to Muon by the
        # >=2D rule. It is a norm gain, not a linear map -- see the note in focus_compete.py.
        self.adam_scale = nn.Parameter(
            torch.ones(self.n_focus, self.channels, dtype=dtype)
        )

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, C); also accepts (B, C) when `n_focus=1`. Same shape / dtype out."""
        in_dtype = x.dtype
        x = x.to(dtype=self.compute_dtype)
        eps = torch.tensor(self.eps, dtype=x.dtype, device=x.device)

        if x.ndim == 2:
            inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
            x = x * inv_rms
            x = x * self.adam_scale[0]
            return x.to(dtype=in_dtype)

        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        x = x * inv_rms
        x = x * self.adam_scale.unsqueeze(0)
        return x.to(dtype=in_dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, n_focus={self.n_focus}, eps={self.eps}"
