"""Four-rank gloo check that native trainer attention with 2 KV heads at TP4 matches TP1.

Each rank holds a quarter of each KV head and ``kv_gather`` reassembles whole heads. Outputs,
input gradients, and gathered weight gradients must match the TP1 component; replicated q/k
norm gradients are summed once over the TP group.
"""

from __future__ import annotations

import itertools

import torch
from torch import fx, nn

from isoexec.ops.collectives.tests._gloo_harness import assert_claim, run_world

WORLD = 4
CU = [0, 8, 20]
SEED = 23


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).abs().max() / expected.abs().max())


def _install_cpu_gemm() -> None:
    """Substitute pik's CUDA GEMM and symmetric-memory staging with CPU equivalents; the row-parallel
    reduction tree and autograd stay production."""
    import sys
    import types

    from isoexec.ops.collectives.pik import allreduce
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    gemm = types.ModuleType("isoexec.ops.collectives.pik.gemm")

    def ti_gemm(x, w, *, plan, tp_size, tp_rank, k_full, out=None):
        part = torch.mm(x.float(), w.float().t())
        return part if out is None else out.copy_(part)

    gemm.ti_gemm = ti_gemm
    sys.modules[gemm.__name__] = gemm
    from isoexec.ops.collectives.pik import linear

    allreduce.p2p_available = linear.p2p_available = lambda group=None: False
    ensure_pik()


HIDDEN = 256
HEAD_DIM = 64


def _attention(*, device: str = "meta"):
    """Eight query heads over two KV heads: at TP4 each rank holds half a KV head."""
    from isoexec.nn import Attention

    return Attention(hidden=HIDDEN, heads=8, kv_heads=2, head_dim=HEAD_DIM, rope_theta=1e7, rotary_fraction=0.25,
                     qk_norm="rms_zero_centered", eps=1e-6, out_gate="sigmoid", device=device)


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch.distributed as dist

    from isoexec.core.arch import NON_ACCELERATOR_ARCH
    from isoexec.compiler.realization.storage import allocate_tensors
    from isoexec.compiler.realization.tp import gather_weight, set_local_dims, shard_parameter_shapes, slice_weight
    from isoexec.nn import place
    from isoexec.ops import _dispatch
    from isoexec.ops.collectives.tests.test_tp_bytes_cpu import _install_cpu_shims
    from isoexec.ops.table import OpTable
    from isoexec.runtimes.parallel import trivial_placement
    from isoexec.runtimes.megatron._binding import bind_trainer_ops
    from isoexec.schema import make_sequence_context

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    _install_cpu_shims()
    _install_cpu_gemm()
    group = dist.group.WORLD
    reference = _attention(device="cpu")
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=gen).to(parameter.dtype))

    root = nn.Module()
    root.layer = _attention()
    graph = fx.Graph()
    graph.output(graph.call_module("layer", (graph.placeholder("x"), graph.placeholder("seq"))))
    native = fx.GraphModule(root, graph)
    shard_parameter_shapes(native, tp_size=world, side="trainer")
    set_local_dims(native, tp_size=world)
    # Production collectives are declared for sm90; their gloo bodies run on CPU.
    _dispatch.ARCH = "sm90"
    bind_trainer_ops(native, tp_group=group, op_table=OpTable(mode="reference"))
    _dispatch.ARCH = NON_ACCELERATOR_ARCH
    place(reference, trivial_placement(), prefix="lm.layers.0.mixer", op_table=OpTable(mode="reference"))
    allocate_tensors(native, device=torch.device("cpu"))
    full = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in native.layer.named_parameters():
            parameter.copy_(slice_weight(native, f"layer.{name}", full[name], tp_size=world, tp_rank=rank,
                                         side="trainer"))

    x = torch.randn(CU[-1], HIDDEN, generator=gen).to(torch.bfloat16)
    upstream = torch.randn(CU[-1], HIDDEN, generator=gen).to(torch.bfloat16)
    positions = torch.cat([torch.arange(b - a) for a, b in itertools.pairwise(CU)]).to(torch.int64)
    seq = make_sequence_context(positions, torch.tensor(CU, dtype=torch.int32),
                                max_q=max(b - a for a, b in itertools.pairwise(CU)))
    x_ref, x_native = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    expected, actual = reference(x_ref, seq), native.layer(x_native, seq)
    (expected.float() * upstream.float()).sum().backward()
    (actual.float() * upstream.float()).sum().backward()

    layer = native.layer
    res = {
        "kv_gather_is_bound": layer.slots.resolved()["kv_gather"].impl == "nccl_bytes_tree_rs_bwd",
        "local_heads": (layer.heads_local, layer.kv_heads_local) == (2, 1),
        "kv_rows_are_a_head_fraction": tuple(layer.k.shape) == (2 * HEAD_DIM // world, HIDDEN),
        "output": _relative_error(actual, expected),
        "input_grad": _relative_error(x_native.grad, x_ref.grad),
    }
    errors = {}
    for name, parameter in layer.named_parameters():
        grad = gather_weight(native, f"layer.{name}", parameter.grad, tp_group=group, side="trainer")
        errors[name] = _relative_error(grad, full[name].grad)
    res["weight_grads"] = errors
    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


def test_native_tp4_attention_matches_tp1_with_split_kv_heads(tmp_path):
    results = run_world(_body, world=WORLD, rendezvous=tmp_path / "store")
    for claim in ("kv_gather_is_bound", "local_heads", "kv_rows_are_a_head_fraction"):
        assert_claim(results, claim, WORLD)
    for rank, res in results.items():
        # BF16 row-parallel and fixed-tree sums differ from one TP1 GEMM by rounding only; a
        # misplaced head window or a missing or duplicated q/k norm reduction is an O(1) error.
        assert res["output"] < 2e-2 and res["input_grad"] < 2e-2, (rank, res["output"], res["input_grad"])
        assert set(res["weight_grads"]) >= {"q", "k", "v", "o", "q_norm", "k_norm"}, rank
        assert max(res["weight_grads"].values()) < 2e-2, (rank, res["weight_grads"])
