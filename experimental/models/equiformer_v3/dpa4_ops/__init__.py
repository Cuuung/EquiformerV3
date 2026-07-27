"""DPA4 (deepmd-kit fork, internal name SeZM) operators ported for ablation on EquiformerV3.

Everything here is OFF by default. The three switches, all in the yml `model:` block:

    envelope_type:          'equiformerv3_c2'   # D4: -> 'dpa4_c3' for the C3 cutoff envelope
    attn_softmax_type:      'equiformerv3'      # D1: -> 'dpa4_envelope_gated'
    focus_compete_groups:   0                   # D2: -> 2 / 4 / 8 (must divide num_heads)

With the defaults the forward pass is BIT-IDENTICAL to the pre-change model and the state_dict
key set is unchanged; `tests/models/equiformer_v3/test_dpa4_switches.py` pins that.

Source of the ports (read-only reference):
    ../DPA4/deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/
    commit 99c1ece2e5087c77267fba4ca84932b53621e42c
routed through mlip-forge's `_dpa4_ops`, whose parity tests were run against the real DPA4 clone.
Per-file provenance headers are kept in each module.

WARNING -- new parameters are NOT protected by the checkpoint loader. `load_pretrained_weights`
in `equiformer_v3_dens_trainer.py` builds its dict FROM the model and only overwrites keys the
checkpoint happens to contain, so `strict=True` can never fire: parameters that exist in the
model but not in the checkpoint are silently left at their random init, with no log line. D1's
`z_bias_raw` and D2's `focus_compete.*` are exactly such parameters. Therefore the direct and
grad-finetune stages MUST be switched on/off together. `check_dpa4_switch_transfer()` below is
provided so a run can assert that explicitly.
"""
from __future__ import annotations

from .attention import segment_envelope_gated_softmax
from .focus_compete import CrossFocusCompetition
from .norm import ScalarRMSNorm
from .softmax_adapter import EnvelopeGatedGraphSoftmax

__all__ = [
    "segment_envelope_gated_softmax",
    "EnvelopeGatedGraphSoftmax",
    "CrossFocusCompetition",
    "ScalarRMSNorm",
    "dpa4_parameter_names",
    "check_dpa4_switch_transfer",
]

#: Suffixes of every parameter introduced by the DPA4 switches. Used to (a) route them off Muon
#: and (b) detect a stage-1 -> stage-2 mismatch.
DPA4_PARAM_SUFFIXES = (
    "z_bias_raw",                                # D1
    "focus_compete.adamw_focus_compete_w",       # D2
    "focus_compete.focus_compete_bias",          # D2 (only when bias=True)
    "focus_compete.focus_compete_norm.adam_scale",  # D2
)


def dpa4_parameter_names(model) -> list[str]:
    """Names of all DPA4-introduced parameters currently present in `model`."""
    return [
        name
        for name, _ in model.named_parameters()
        if any(name.endswith(s) for s in DPA4_PARAM_SUFFIXES)
    ]


def check_dpa4_switch_transfer(model, checkpoint_state_dict) -> list[str]:
    """Return DPA4 parameters that the checkpoint will NOT supply (i.e. stay at random init).

    An empty list means every DPA4 parameter in `model` is covered by the checkpoint. A non-empty
    list on a continuation run means the switches were NOT set identically in the two stages --
    the run would silently train from a half-initialised model.
    """
    ckpt_keys = set()
    for k in checkpoint_state_dict:
        for prefix in ("_orig_mod.module.", "module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        ckpt_keys.add(k)
    return [n for n in dpa4_parameter_names(model) if n not in ckpt_keys]
