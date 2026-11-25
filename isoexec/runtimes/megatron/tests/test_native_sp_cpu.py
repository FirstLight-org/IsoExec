"""Two-rank gloo check that native trainer stages with SP match the same model at TP2 without SP.

The model is a captured attention/MLP Qwen3.5 model, run as one stage and as two PP stages chained
in process, where the first stage's token chunk is the second stage's ``hidden_states``. Under SP the
residual stream holds each rank's token chunk, so logits and every gradient except the RMSNorm
weights' must be bitwise equal to TP2 without SP. RMSNorm weight gradients are partial per chunk;
after the TP sum Megatron's finalize applies to ``sequence_parallel`` Parameters they match up to
summation order.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.collectives.tests._gloo_harness import assert_claim, run_world

WORLD = 2
CU = [0, 4, 12]
SEED = 31


def _stage(assignments=((0, 2),), pp_rank=0):
    from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.text import text_inputs
    from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    config = qwen_config(tied=False)
    config.text_config.layer_types = ["full_attention", "full_attention"]
    author = Qwen3_5ForConditionalGeneration(config).train(True)
    inputs = text_inputs(author, device="cpu", max_tokens=16, profile="trainer")
    captured = capture(author, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    return define_stage(captured, assignments, pp_rank=pp_rank)


def _table():
    from dataclasses import replace

    from isoexec.ops.table import OpTable

    class Table(OpTable):
        """The stage forward performs no logprob math; the head's required TP seam is explicit."""

        def bind(self, op, **kwargs):
            binding = super().bind(op, **kwargs)
            if op != "logprobs.log_softmax":
                return binding

            def unexecuted(*_args, group=None):
                raise AssertionError("The stage forward must not execute log_softmax")
            return replace(binding, fn=unexecuted, impl="cpu_fixture.unexecuted_log_softmax")

    return Table(mode="reference")


def _build(definition, group, *, sp: bool, pp: bool = False):
    from isoexec.core.arch import NON_ACCELERATOR_ARCH
    from isoexec.compiler.realization.graph import clone_graph_module, drop_auxiliary_outputs
    from isoexec.compiler.realization.pp import prepare_pipeline_graph
    from isoexec.compiler.realization.sp import prepare_sequence_parallel_graph
    from isoexec.compiler.realization.storage import allocate_tensors
    from isoexec.compiler.realization.tp import set_head_output_shape, set_local_dims, shard_parameter_shapes
    from isoexec.ops import _dispatch
    from isoexec.ops.collectives.tp_bytes import bind_seq_scatter
    from isoexec.runtimes.megatron._binding import bind_trainer_ops, set_sequence_parallel_attributes

    module = clone_graph_module(definition.module)
    drop_auxiliary_outputs(module)
    if pp:
        prepare_pipeline_graph(module)
    shard_parameter_shapes(module, tp_size=WORLD, side="trainer")
    set_local_dims(module, tp_size=WORLD)
    set_head_output_shape(module)
    # Production collectives are declared for sm90; their gloo bodies run on CPU.
    _dispatch.ARCH = "sm90"
    if sp:
        scatter = bind_seq_scatter(tp_group=group, site="seq_scatter")
        prepare_sequence_parallel_graph(module, tp_size=WORLD, scatter=scatter.fn)
    bind_trainer_ops(module, tp_group=group, sp=sp, op_table=_table())
    _dispatch.ARCH = NON_ACCELERATOR_ARCH
    allocate_tensors(module, device=torch.device("cpu"))
    if sp:
        set_sequence_parallel_attributes(module)
    return module


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch.distributed as dist
    from megatron.core.tensor_parallel import random

    from isoexec.runtimes.megatron._recompute import checkpoint_layers

    # Keep MCore's checkpoint/autograd implementation; substitute CPU RNG I/O.
    random._get_all_rng_states = lambda: (torch.get_rng_state(), None, None)
    random._set_all_rng_states = lambda cpu, cuda, tracker: torch.set_rng_state(cpu)

    from isoexec.compiler.realization.tp import gather_weight, slice_weight
    from isoexec.ops.collectives.tests.test_tp_bytes_cpu import _install_cpu_shims
    from isoexec.runtimes.megatron._inputs import packed_sequence_context
    from isoexec.runtimes.megatron.tests.test_native_tp_attention_cpu import _install_cpu_gemm

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    _install_cpu_shims()
    _install_cpu_gemm()
    group = dist.group.WORLD
    definition = _stage()
    gen = torch.Generator().manual_seed(SEED)
    full = {name: torch.randn(parameter.shape, generator=gen).to(parameter.dtype)
            for name, parameter in definition.module.named_parameters()}
    vocab = full["head.weight"].shape[0]
    ids = torch.randint(0, vocab, (1, CU[-1]), generator=gen)
    upstream = torch.randn(CU[-1], vocab, generator=gen)

    cu = torch.tensor(CU, dtype=torch.int32)
    longest = max(b - a for a, b in zip(CU, CU[1:]))

    def context(token_ids):
        # Each forward builds its own context, as MegatronModel does; a forward caches rotary state in it.
        return packed_sequence_context(token_ids, None, cu, max_q=longest, max_kv=None, mrope=True)

    def load(module):
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.copy_(slice_weight(module, name, full[name], tp_size=world, tp_rank=rank, side="trainer"))
        return module

    def finish(modules, local):
        width = local.shape[-1]
        (local.float() * upstream[:, rank * width:(rank + 1) * width]).sum().backward()
        named = [(name, parameter, module) for module in modules for name, parameter in module.named_parameters()]
        for _, parameter, _ in named:
            if getattr(parameter, "sequence_parallel", False):
                # Megatron's finalize_model_grads sums these over TP when config.sequence_parallel is set.
                dist.all_reduce(parameter.grad, group=group)
        grads = {name: gather_weight(module, name, parameter.grad, tp_group=group, side="trainer")
                 for name, parameter, module in named}
        marked = sorted(name for name, parameter, _ in named if getattr(parameter, "sequence_parallel", False))
        return local.detach(), grads, marked

    runs, res = {}, {}
    for name, sp in (("tp2", False), ("sp", True), ("sp_checkpoint", True), ("sp_checkpoint_span", True)):
        module = load(_build(definition, group, sp=sp))
        if name.startswith("sp_checkpoint"):
            checkpoint_layers(module, span=2 if name.endswith("span") else 1)
        seq = context(ids)
        runs[name] = finish([module], module(input_ids=seq.input_ids, seq=seq))
        if sp:
            res["scatter_once"] = sum(node.op == "call_function" and node.target.__name__.startswith("seq_scatter")
                                      for node in module.graph.nodes) == 1
    first, last = (load(_build(_stage(((0, 1), (1, 2)), stage), group, sp=True, pp=True)) for stage in range(2))
    seq = context(ids)
    chunk = first(input_ids=seq.input_ids, seq=seq, hidden_states=None)
    res["pp_chunk_rows"] = chunk.shape[0] == CU[-1] // world
    runs["sp_pp2"] = finish([first, last], last(input_ids=None, seq=context(None), hidden_states=chunk))

    first, last = (load(_build(_stage(((0, 1), (1, 2)), stage), group, sp=True, pp=True)) for stage in range(2))
    checkpoint_layers(first)
    checkpoint_layers(last)
    seq = context(ids)
    chunk = first(input_ids=seq.input_ids, seq=seq, hidden_states=None)
    runs["sp_pp2_checkpoint"] = finish([first, last], last(input_ids=None, seq=context(None), hidden_states=chunk))

    logits, grads, _ = runs["tp2"]
    for name in ("sp", "sp_pp2", "sp_checkpoint", "sp_checkpoint_span", "sp_pp2_checkpoint"):
        actual, actual_grads, marked = runs[name]
        res[name] = {
            "logits_equal": torch.equal(actual, logits),
            "marked_norms": marked,
            "inexact": sorted(key for key in grads if not torch.equal(actual_grads[key], grads[key])),
            "norm_errors": {key: float((actual_grads[key].float() - grads[key].float()).abs().max()
                                       / grads[key].float().abs().max()) for key in marked},
        }
    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_world(_body, world=WORLD, rendezvous=tmp_path_factory.mktemp("sp") / "store")


@pytest.mark.parametrize("layout", ["sp", "sp_pp2", "sp_checkpoint", "sp_checkpoint_span", "sp_pp2_checkpoint"])
def test_native_sp_matches_tp2_without_sp(results, layout):
    for claim in ("scatter_once", "pp_chunk_rows"):
        assert_claim(results, claim, WORLD)
    norms = {"layers.0.attn_norm.weight", "layers.0.mlp_norm.weight", "layers.1.attn_norm.weight",
             "layers.1.mlp_norm.weight", "norm.weight"}
    for rank, res in results.items():
        run = res[layout]
        assert run["logits_equal"], rank
        assert set(run["marked_norms"]) == norms and set(run["inexact"]) <= norms, (rank, run["inexact"])
        # A missing or duplicated TP sum is an O(1) error; summation order is bf16 rounding.
        assert max(run["norm_errors"].values()) < 2e-2, (rank, run["norm_errors"])


def test_native_sp_refuses_an_unsupported_operation_on_the_token_chunk():
    import operator

    from isoexec.compiler.realization.graph import clone_graph_module
    from isoexec.compiler.realization.sp import prepare_sequence_parallel_graph

    module = clone_graph_module(_stage().module)
    add = next(node for node in module.graph.nodes if node.target is torch.ops.aten.add.Tensor)
    with module.graph.inserting_after(add):
        size = module.graph.call_function(torch.ops.aten.sym_size.int, (add, 0))
        module.graph.call_function(operator.add, (size, 1))
    with pytest.raises(ValueError, match="sym_size"):
        prepare_sequence_parallel_graph(module, tp_size=WORLD, scatter=lambda x: x)
