"""REGRESSION: no DPA4 switch may bake a concrete SHAPE into the traced graph.

Why this file exists
--------------------
The first D1 grad-finetune submission died in the conservative-force compile path with

    Dynamo failed to run FX node with fake tensors: aten.add.Tensor(
        FakeTensor(size=(7, 9, 64)), FakeTensor(size=(s77, 9, 64)))
    RuntimeError('The size of tensor a (7) must match the size of tensor b (s77: hint = 431)')

Root cause: the D1 adapter did `n_nodes = int(num_nodes)`. The conservative path make_fx-traces
the model with a dummy prime-shaped example (`compile_utils.make_prime_graph_example`, natoms=7,
nedges=11). Calling `int()` on a symbolic size MATERIALISES the trace-time value, so the traced
graph carried a literal 7 for the node dimension while everything else stayed symbolic. The
number 7 in the error is that dummy natoms, exactly.

What makes this class of bug nasty: it is invisible in eager mode (all tests passed, both direct
stages trained for 15 epochs), and it only detonates in the conservative grad-ft stage -- i.e.
after a full direct run has already been spent.

This test reproduces the mechanism WITHOUT needing the pool's torch 2.11: symbolic `make_fx` is
available in the dev torch too. Any future switch that ints/items a shape fails here immediately.

Run:  PYTHONPATH=src ./.venv/bin/python -m pytest tests/experimental/equiformer_v3/test_dpa4_symbolic_trace.py -q
"""
from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from fairchem.experimental.models.equiformer_v3.dpa4_ops import EnvelopeGatedGraphSoftmax
from fairchem.experimental.models.equiformer_v3.softmax import GraphSoftmax

# The dummy trace shapes the conservative path actually uses (compile_utils:209).
TRACE_NODES, TRACE_EDGES = 7, 11
# A "real" batch, deliberately unrelated to the trace shapes.
REAL_NODES, REAL_EDGES = 431, 977
N_HEADS = 8


def _run(mod, n_nodes, n_edges, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n_edges, N_HEADS, generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    env = torch.rand(n_edges, 1, generator=g)
    return mod(logits, index=dst, num_nodes=n_nodes, exp_rescale=env)


def _inputs(n_nodes, n_edges, seed):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n_edges, N_HEADS, generator=g),          # logits
        torch.randint(0, n_nodes, (n_edges,), generator=g),  # dst
        torch.rand(n_edges, 1, generator=g),                 # envelope
        torch.zeros(n_nodes),                                # stands in for node features x
    )


def _trace_symbolic(mod, n_nodes, n_edges):
    """make_fx in symbolic mode -- the same mode the dynamic conservative path uses.

    Two things this harness must get right or the test passes vacuously:
      * `num_nodes` comes from a TRACED TENSOR's shape (`node_ref.shape[0]`), never a closed-over
        Python int -- the real model passes `x.shape[0]`, which is exactly this.
      * module parameters are threaded in as explicit inputs via `functional_call`, otherwise
        fake-tensor mode rejects the real `z_bias_raw` Parameter.
    """
    names = list(dict(mod.named_parameters()))

    def fn(logits, dst, env, node_ref, *pvals):
        return torch.func.functional_call(
            mod, dict(zip(names, pvals)),
            args=(logits,),
            kwargs=dict(index=dst, num_nodes=node_ref.shape[0], exp_rescale=env),
        )

    pvals = tuple(p.detach().clone() for p in mod.parameters())
    return make_fx(fn, tracing_mode="symbolic")(*_inputs(n_nodes, n_edges, 0), *pvals), names


def _run_traced(gm_names, mod, n_nodes, n_edges, seed=1):
    gm, _ = gm_names
    ins = _inputs(n_nodes, n_edges, seed)
    pvals = tuple(p.detach().clone() for p in mod.parameters())
    return gm(*ins, *pvals), ins


@pytest.mark.parametrize(
    "make_mod",
    [lambda: GraphSoftmax(), lambda: EnvelopeGatedGraphSoftmax(num_heads=N_HEADS)],
    ids=["baseline", "dpa4_envelope_gated"],
)
def test_graph_has_no_hardcoded_trace_node_count(make_mod):
    """The literal 7 (dummy natoms) must NOT appear as a size in the traced graph.

    This is THE assertion that would have caught the failed submission before it was launched.
    """
    gm, _ = _trace_symbolic(make_mod(), TRACE_NODES, TRACE_EDGES)
    code = gm.code
    offenders = [
        line
        for line in code.splitlines()
        # a baked node-dim shows up as a literal in a shape-taking op
        if any(op in line for op in ("zeros", "full", "empty", "new_zeros"))
        and f"{TRACE_NODES}," in line.replace(" ", "")
    ]
    assert not offenders, (
        "a concrete node count was baked into the traced graph:\n  "
        + "\n  ".join(offenders)
        + "\n\nThis is the int(num_nodes) failure mode: the graph traced with the dummy "
          "natoms=7 example will not run on a real batch. Keep shapes symbolic "
          "(use maybe_num_nodes, never int()/item() on a size)."
    )


@pytest.mark.parametrize(
    "make_mod",
    [lambda: GraphSoftmax(), lambda: EnvelopeGatedGraphSoftmax(num_heads=N_HEADS)],
    ids=["baseline", "dpa4_envelope_gated"],
)
def test_traced_graph_generalises_to_a_different_batch_size(make_mod):
    """Trace at the dummy shapes, then RUN at real shapes -- the actual production sequence."""
    mod = make_mod()
    gm = _trace_symbolic(mod, TRACE_NODES, TRACE_EDGES)
    # would raise the (7) vs (431) size error before the fix
    out, _ = _run_traced(gm, mod, REAL_NODES, REAL_EDGES)
    assert out.shape == (REAL_EDGES, N_HEADS)
    assert torch.isfinite(out).all()


def test_dpa4_softmax_normalisation_survives_the_shape_change():
    """Sanity: the traced graph must still compute the DPA4 formula, not just any finite tensor."""
    mod = EnvelopeGatedGraphSoftmax(num_heads=N_HEADS)
    gm = _trace_symbolic(mod, TRACE_NODES, TRACE_EDGES)
    traced, (logits, dst, env, _n) = _run_traced(gm, mod, REAL_NODES, REAL_EDGES, seed=2)
    eager = mod(logits, index=dst, num_nodes=REAL_NODES, exp_rescale=env)
    assert torch.allclose(traced, eager, atol=1e-6), "traced graph diverges from eager"


def test_no_int_or_item_on_shapes_in_dpa4_sources():
    """Literal source guard: the pattern that caused the failure must not come back."""
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[3] / "experimental" / "models" / "equiformer_v3" / "dpa4_ops"
    banned = []
    for f in pkg.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for pat in ("int(num_nodes", "int(n_nodes", ".item()", "int(index.max"):
                if pat in s:
                    banned.append(f"{f.name}:{i}: {s}")
    assert not banned, (
        "shape-materialising call re-introduced in dpa4_ops:\n  " + "\n  ".join(banned)
    )
