"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Model-agnostic make_fx + torch.compile harness for compiling the *conservative*
(auto_grad) force path, where ``forces = -autograd.grad(E, pos, create_graph=...)``
introduces a double-backward that plain ``torch.compile`` cannot trace.

Blueprint (verified):
  * deepmd-kit DPA4 / SeZM ``deepmd/pt/model/model/sezm_model.py`` (``core_compute``,
    ``_remove_detach_nodes``, silu_backward decomposition, prime-dim trick).
  * Local PoC ``phase2_makefx_e2e.py`` — proved this exact pipeline reproduces
    eager forces (1e-8) and weight gradients (1e-9) on eSEN's auto_grad path.

Pipeline:
    pos.requires_grad -> core_compute (pure tensor in/out) -> energy
    -> forces = -autograd.grad(E, pos, create_graph=True)   (double backward)
    -> make_fx(real) traces the whole thing (fwd + both backwards) into aten ops
    -> strip_detach    : detach nodes silently sever the 2nd-order grad path; remove them
    -> rebuild_graph   : fresh GraphModule so inductor sees a clean graph
    -> torch.compile(inductor) with silu_backward decomposed
    -> training step: loss(forces).backward() now flows to weights.

This module is intentionally model-agnostic: it operates on an arbitrary
``core_fn`` (a pure tensor-in/tensor-out closure) and a tuple of example inputs.
eSEN and EquiformerV3 share it; each model only provides its own ``core_fn`` and
keeps neighbor-list / graph construction in the eager region (data-dependent
control flow does not belong in the traced region).

The two force modes map onto two strategies that *share the same core_compute*:
  * direct force  -> plain ``torch.compile`` (no double backward); see ``plain_compile``.
  * conservative  -> this make_fx harness; see ``trace_and_compile``.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, Sequence

import torch
import torch.fx as fx


# --------------------------------------------------------------------------- #
# dynamo / ddp configuration
# --------------------------------------------------------------------------- #
def configure_dynamo_for_compile(optimize_ddp: bool = False) -> None:
    """Set the dynamo flags required for compiling an *inner* region under DDP.

    ``optimize_ddp=False`` disables DDPOptimizer, whose graph splitting crashes
    when we compile a region nested inside the model (DDP itself still works,
    gradients still all-reduce). deepmd uses the same setting
    (``pt_expt/train/training.py:1140``). Idempotent; safe to call every step.
    """
    torch._dynamo.config.optimize_ddp = optimize_ddp
    # 探针②：ESEN_COMPILE_PROBE=1 时禁止编译失败静默退回 eager —— 失败直接抛，
    # 用来确认「真的在编译」。默认不动 dynamo 行为。
    if os.environ.get("ESEN_COMPILE_PROBE", "0") == "1":
        torch._dynamo.config.suppress_errors = False
    # EQV3_ACT_MEM_BUDGET：AOTAutograd min-cut partitioner 的 activation_memory_budget
    # （默认 1.0=runtime 最优、几乎不重算）。调低（如 0.6/0.5）让 partitioner 解 0-1 背包、
    # 用重算换显存，治保守力 make_fx 图的 saved-tensor 膨胀。等价于 esen 侧的同名旋钮，故也接受
    # ESEN_ACT_MEM_BUDGET 别名。torch 自校验范围 [0,1]。不设则不碰（no-op）。
    _amb = os.environ.get("EQV3_ACT_MEM_BUDGET") or os.environ.get("ESEN_ACT_MEM_BUDGET")
    if _amb is not None:
        import torch._functorch.config as _ft_config

        _ft_config.activation_memory_budget = float(_amb)


# --------------------------------------------------------------------------- #
# decompositions
# --------------------------------------------------------------------------- #
def get_force_decompositions(extra: Iterable | None = None):
    """Decomposition table needed for double-backward force graphs.

    ``aten.silu_backward`` (and friends) appear in the *backward* of the energy
    network; under ``create_graph=True`` they must be decomposed into primitive
    ops for inductor to differentiate them again. Mirrors deepmd's
    ``sezm_model.py`` NOTE 2 and the verified PoC.
    """
    from torch._decomp import get_decompositions

    targets = [
        torch.ops.aten.silu_backward.default,
    ]
    if extra is not None:
        targets.extend(extra)
    return get_decompositions(targets)


# --------------------------------------------------------------------------- #
# fx graph surgery
# --------------------------------------------------------------------------- #
def strip_detach(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove ``aten.detach`` nodes that make_fx inserts around the autograd.grad
    boundary.

    These detaches silently sever the second-order gradient path: forces compute
    correctly but ``loss(forces).backward()`` produces no weight gradients, with
    *no error*. This is the single most dangerous silent failure of the whole
    pipeline (roadmap §5), so the integration gate requires "force RMSE actually
    decreases". Equivalent to deepmd ``_remove_detach_nodes``.

    Mutates ``gm`` in place and returns it.
    """
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def replace_view_with_reshape(gm: fx.GraphModule) -> fx.GraphModule:
    """Normalize ``aten.view`` -> ``aten.reshape`` in a make_fx graph.

    make_fx records ``aten.view`` ops that are valid on the *real* strides seen
    at trace time. When inductor re-traces the rebuilt graph with fake tensors it
    can recompute strides differently (e.g. a ``view`` right after a ``permute``)
    and raise "Cannot view a tensor with shape ... as ...". ``reshape`` is the
    semantics-preserving equivalent — it returns a view when the layout allows
    and copies otherwise — so this pass removes the brittle stride dependency
    without changing results. Mutates ``gm`` in place and returns it.
    """
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target in (
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        ):
            node.target = torch.ops.aten.reshape.default
    gm.graph.lint()
    gm.recompile()
    return gm


def rebuild_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """Return a fresh GraphModule by copying every node into a new Graph.

    After ``strip_detach`` mutates the graph, inductor traces more reliably on a
    cleanly rebuilt graph than on the edited one (observed in the PoC and in
    deepmd). Parameters/buffers are inherited from ``gm``.
    """
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for node in gm.graph.nodes:
        env[node] = new_graph.node_copy(node, lambda n: env[n])
    new_graph.lint()
    return fx.GraphModule(gm, new_graph)


# --------------------------------------------------------------------------- #
# trace + compile
# --------------------------------------------------------------------------- #
def make_fx_trace(
    core_fn: Callable[..., torch.Tensor],
    example_inputs: Sequence[torch.Tensor],
    *,
    decompositions=None,
    tracing_mode: str = "real",
    allow_non_fake_inputs: bool = True,
) -> fx.GraphModule:
    """make_fx-trace ``core_fn(*example_inputs)`` into an aten GraphModule, then
    strip detaches and rebuild.

    ``core_fn`` must be a pure tensor-in / tensor-out closure (any non-traced
    state — neighbor lists, cell offsets, weights — captured in the closure).
    ``tracing_mode="real"`` is used because eSEN/SeZM's symbolic mode hits
    ``Eq(u0,1)`` shape collisions on edge counts; the prime-dim trick (roadmap
    NOTE 3) is the path to symbolic/dynamic tracing and is layered on top later.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    if decompositions is None:
        decompositions = get_force_decompositions()

    gm = make_fx(
        core_fn,
        tracing_mode=tracing_mode,
        _allow_non_fake_inputs=allow_non_fake_inputs,
        decomposition_table=decompositions,
    )(*example_inputs)
    gm = strip_detach(gm)
    gm = replace_view_with_reshape(gm)
    gm = rebuild_graph(gm)
    return gm


# Inductor options locked for the dynamic=True conservative path. mix_order /
# persistent reductions and autotune/epilogue fusion are the knobs that miscompile
# (or recompile-per-shape) the double-backward symbolic graph on torch 2.11; this
# set is the one verified PASS in compile_dev/probe_symbolic_compile.py. Filtered
# to keys that exist in the running inductor build before use.
LOCKED_INDUCTOR_OPTIONS: dict = {
    "max_autotune": False,
    "shape_padding": True,
    "epilogue_fusion": False,
    "triton.cudagraphs": False,
    "max_fusion_size": 8,
    "triton.persistent_reductions": False,
    "triton.mix_order_reduction": False,
}


def make_prime_graph_example(
    device, dtype, *, stress: bool, nsys: int = 5, natoms: int = 7, nedges: int = 11
) -> tuple:
    """Pairwise-distinct prime dims (deepmd NOTE 1) for the symbolic make_fx of the
    conservative-force ``core_fn``. Real composite shapes (natoms=64=2^6,
    nedges=5120) get factored by make_fx's symbolic shape inference and unified
    with static dims (3/9/channels) or constants, leaking concrete sizes into
    guards so every shape recompiles; distinct primes >=5 stay genuinely symbolic.
    Layout matches eSEN's core_fn: ``(pos[, disp], an, ei, co, cell, batch)``.
    """
    pos = torch.randn(natoms, 3, device=device, dtype=dtype, requires_grad=True)
    an = torch.randint(1, 90, (natoms,), device=device)
    ei = torch.randint(0, natoms, (2, nedges), device=device)
    co = torch.randn(nedges, 3, device=device, dtype=dtype)
    cell = torch.randn(nsys, 3, 3, device=device, dtype=dtype) + torch.eye(
        3, device=device
    ) * 5
    batch = torch.arange(natoms, device=device) % nsys  # every system non-empty
    if stress:
        disp = torch.zeros(
            nsys, 3, 3, device=device, dtype=dtype, requires_grad=True
        )
        return (pos, disp, an, ei, co, cell, batch)
    return (pos, an, ei, co, cell, batch)


def _filter_inductor_options(options: dict) -> dict:
    """Keep only keys present in the running inductor config (key set varies
    across torch versions; an unknown key raises)."""
    try:
        from torch._inductor import config as ic

        keys = ic.get_config_copy()
        return {k: v for k, v in options.items() if k in keys}
    except Exception:  # noqa: BLE001
        return dict(options)


def trace_and_compile(
    core_fn: Callable[..., torch.Tensor],
    example_inputs: Sequence[torch.Tensor],
    *,
    backend: str = "inductor",
    dynamic: bool = False,
    decompositions=None,
    inductor_options: dict | None = None,
    optimize_ddp: bool = False,
    reset_dynamo: bool = True,
    tracing_mode: str = "real",
) -> Callable:
    """Full conservative-force harness: make_fx -> strip_detach -> rebuild ->
    torch.compile.

    Returns a compiled callable with the same signature as the traced
    ``core_fn`` (i.e. takes the dynamic tensors that were passed as
    ``example_inputs``). The returned graph is specialized to the *shapes* of
    ``example_inputs`` when ``dynamic=False``; callers that see varying system
    sizes must re-trace (use :class:`CompiledForceRegion`) or move to
    ``dynamic=True`` once the prime-dim trick is in place.

    Args mirror the verified PoC (``phase2_makefx_e2e.py``); ``dynamic=False``
    is the proven default on torch 2.4 (dynamic=True hits an inductor codegen
    bug on otf_graph edge counts — see roadmap §1.6).
    """
    configure_dynamo_for_compile(optimize_ddp=optimize_ddp)
    gm = make_fx_trace(
        core_fn, example_inputs, decompositions=decompositions, tracing_mode=tracing_mode
    )

    if reset_dynamo:
        torch._dynamo.reset()

    compile_kwargs: dict = {"backend": backend, "dynamic": dynamic}
    if inductor_options is not None:
        compile_kwargs["options"] = inductor_options
    return torch.compile(gm, **compile_kwargs)


def plain_compile(
    fn: Callable,
    *,
    backend: str = "inductor",
    dynamic: bool = False,
    fullgraph: bool = False,
    optimize_ddp: bool = False,
    options: dict | None = None,
) -> Callable:
    """Strategy for the *direct-force* path (no double backward): just
    ``torch.compile`` with the DDP-safe dynamo flag set.

    Verified at 1.26–1.79x on eSEN's direct_force forward+backward (roadmap §1.6).
    Shares ``core_compute`` with the conservative path; only the wrapper differs.
    """
    configure_dynamo_for_compile(optimize_ddp=optimize_ddp)
    kwargs: dict = {"backend": backend, "dynamic": dynamic, "fullgraph": fullgraph}
    if options is not None:
        kwargs["options"] = options
    return torch.compile(fn, **kwargs)


# --------------------------------------------------------------------------- #
# lazy re-tracing wrapper (for dynamic=False shape bucketing in integration)
# --------------------------------------------------------------------------- #
class CompiledForceRegion:
    """Caches a ``trace_and_compile`` result per input-shape signature.

    With ``dynamic=False`` the traced graph is shape-specialized, so a changing
    system size needs a re-trace. This wrapper keys compiled graphs by the shapes
    of the dynamic inputs and re-traces on a miss — i.e. shape bucketing as the
    fallback before ``dynamic=True`` lands (roadmap §5 / Stage 3). Model-agnostic:
    the caller supplies a ``core_fn`` builder that closes over the current
    eager-region state.

    Usage (in a model/trainer, conservative path)::

        region = CompiledForceRegion(dynamic=False)
        # each step, after building the eager graph:
        forces = region(build_core_fn(an, edge_index, ...), (pos,))
    """

    def __init__(
        self,
        *,
        backend: str = "inductor",
        dynamic: bool = False,
        inductor_options: dict | None = None,
        optimize_ddp: bool = False,
    ) -> None:
        self.backend = backend
        self.dynamic = dynamic
        # dynamic=True traces symbolic and must lock the inductor knobs that
        # otherwise miscompile/recompile-per-shape the double-backward graph
        # (verified set; see LOCKED_INDUCTOR_OPTIONS). dynamic=False keeps the
        # caller's options (default None = inductor defaults).
        if dynamic and inductor_options is None:
            inductor_options = _filter_inductor_options(LOCKED_INDUCTOR_OPTIONS)
        self.inductor_options = inductor_options
        self.optimize_ddp = optimize_ddp
        self._cache: dict[tuple, Callable] = {}

    @staticmethod
    def _shape_key(example_inputs: Sequence[torch.Tensor]) -> tuple:
        return tuple(
            tuple(t.shape) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        )

    def __call__(
        self,
        core_fn: Callable[..., torch.Tensor],
        example_inputs: Sequence[torch.Tensor],
        *,
        trace_example: Sequence[torch.Tensor] | None = None,
        dynamic_dims: Sequence[tuple[torch.Tensor, Sequence[int]]] | None = None,
    ) -> torch.Tensor:
        """Run ``core_fn(*example_inputs)`` through the compiled region.

        dynamic=False: shape-specialized graph cached per input-shape signature.
        dynamic=True: a single graph is traced *once* in symbolic mode on
        ``trace_example`` (caller-supplied pairwise-distinct prime dims so the
        edge/atom/system sizes stay genuinely symbolic — required, real composite
        shapes get factored and collapse to static guards) and reused for every
        shape. ``dynamic_dims`` lists ``(tensor, [dims])`` of the real inputs to
        ``torch._dynamo.mark_dynamic`` so the first compile builds a dynamic guard.
        """
        if self.dynamic:
            key: tuple = ("dynamic",)
            if dynamic_dims is not None:
                for t, dims in dynamic_dims:
                    for d in dims:
                        torch._dynamo.mark_dynamic(t, d)
        else:
            key = self._shape_key(example_inputs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = trace_and_compile(
                core_fn,
                trace_example if (self.dynamic and trace_example is not None)
                else example_inputs,
                backend=self.backend,
                dynamic=self.dynamic,
                inductor_options=self.inductor_options,
                optimize_ddp=self.optimize_ddp,
                # only reset dynamo on the very first compile to avoid clobbering
                # other cached graphs
                reset_dynamo=(len(self._cache) == 0),
                tracing_mode="symbolic" if self.dynamic else "real",
            )
            self._cache[key] = compiled
            # 探针①：每次 cache-miss = 一次 make_fx+编译。dynamic=True 时
            # cache_size 应停在 1；持续增长 = 重编译风暴(dynamic 没生效)。
            if os.environ.get("ESEN_COMPILE_PROBE", "0") == "1":
                logging.warning(
                    f"[COMPILE] (re)trace -> cache_size={len(self._cache)} "
                    f"dynamic={self.dynamic} key={key}"
                )
        return compiled(*example_inputs)
