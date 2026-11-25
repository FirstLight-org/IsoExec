"""Execute generated owner arithmetic on CPU and compile its SM90 code without a GPU."""

import ast
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.ops.moe.moe_sm90.tests.test_owner_combine_unified_cpu import OC, _fp_ops


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor.reshape(-1), offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def run_emitted(partials, rows, probabilities, shared=None, gate=None):
    """Interpret the actual generated arithmetic with Torch loads/stores, not a duplicate formula."""
    world, _, hidden = partials.shape
    tokens, topk = rows.shape
    size = (tokens + world - 1) // world
    outputs = [torch.empty(size * world, hidden, dtype=torch.bfloat16) for _ in range(world)]
    program = [0, 0]

    def load(pointer, mask=True, other=0.0):
        index = torch.as_tensor(pointer.offset).long()
        valid = torch.as_tensor(mask)
        return torch.where(valid, pointer.tensor[torch.where(valid, index, 0)], other)

    def store(pointer, value, mask=True):
        index = torch.as_tensor(pointer.offset).long()
        valid = torch.broadcast_to(torch.as_tensor(mask), index.shape)
        pointer.tensor[index[valid]] = value[valid]

    tl = SimpleNamespace(
        program_id=lambda axis: program[axis], arange=torch.arange, zeros=torch.zeros,
        static_range=range, minimum=min, load=load, store=store,
        float32=torch.float32, bfloat16=torch.bfloat16, constexpr=int,
    )
    source = OC._unified_push_src(world, topk, False, shared is not None, False, gate is not None, True)
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    fn.decorator_list = []
    namespace = {"tl": tl}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<owner-cpu-emulation>", "exec"), namespace)  # noqa: S102 -- repository-owned generated kernel
    for rank in range(world):
        local_rows, _, padded = OC._engine_my_rows(rows, tokens, topk, world, rank)
        assert padded == size * world
        args = [*(Pointer(value) for value in partials), Pointer(local_rows), Pointer(probabilities)]
        if shared is not None:
            args += [Pointer(value) for value in shared]
            if gate is not None:
                args.append(Pointer(gate))
        args += [Pointer(value) for value in outputs]
        for token in range(size):
            program[0] = token
            namespace["kernel"](*args, rank * size, tokens, hidden, topk, hidden)
    assert all(torch.equal(outputs[0].view(torch.int16), output.view(torch.int16)) for output in outputs)
    return outputs[0][:tokens]


def tree_sum(values):
    if len(values) == 1:
        return values[0].float()
    middle = len(values) // 2
    return tree_sum(values[:middle]) + tree_sum(values[middle:])


@pytest.mark.parametrize("world,topk,tokens", [(2, 1, 3), (2, 3, 4), (8, 1, 5), (8, 8, 9)])
@pytest.mark.parametrize("shared_kind", ["none", "ungated", "gated"])
def test_emitted_owner_matches_complete_tree_then_weight_reference(world, topk, tokens, shared_kind):
    torch.manual_seed(world * 100 + topk)
    hidden = 8
    indices = torch.stack([torch.randperm(16)[:topk] for _ in range(tokens)])
    probs = torch.rand(tokens, topk).float()
    routing = RoutingResult(indices, probs, None)
    # Random inverse permutation; columns remain ascending expert order.
    rows = torch.randperm(tokens * topk).reshape(tokens, topk).int()
    partials = (torch.randn(world, tokens * topk, hidden) * 7).bfloat16().float()
    partials[:, :, 0] = -0.0  # exercise seeded signed-zero behavior
    routed = ExpertOutputs(partials[0], "permuted_partial", routing, combine_rows=rows, weight_after_fc2=True)
    ordered = OC._owner_route_probabilities(routed)
    probabilities = torch.empty(tokens * topk).scatter(0, rows.flatten().long(), ordered.flatten())
    # This is expert_reduce.finish followed by the existing expert-ordered fold.
    weighted = (tree_sum(partials).bfloat16() * probabilities[:, None].bfloat16()).float()
    terms = weighted[rows.long()]
    expected = terms[:, 0].clone()
    for slot in range(1, topk):
        expected = expected + terms[:, slot]
    shared = None if shared_kind == "none" else torch.randn(world, tokens, hidden)
    gate = torch.rand(tokens, 1).bfloat16() if shared_kind == "gated" else None
    if shared is not None:
        shared_result = tree_sum(shared).bfloat16()
        if gate is not None:
            shared_result = shared_result * gate
        expected = expected + shared_result
    actual = run_emitted(partials, rows, ordered, shared, gate)
    assert torch.equal(actual.view(torch.int16), expected.bfloat16().view(torch.int16))
    if topk > 1:
        assert not torch.equal(ordered, probs), "fixture must distinguish expert order from score order"


def test_rounding_before_rank_tree_is_observably_wrong():
    leaves = torch.tensor([1.0, -0.9921875]).view(2, 1, 1).expand(2, 1, 8).clone()
    probabilities = torch.tensor([[0.3333]])
    correct = run_emitted(leaves, torch.tensor([[0]], dtype=torch.int32), probabilities)
    wrong = (leaves.bfloat16() * probabilities.bfloat16()).float().sum(0).bfloat16()
    assert not torch.equal(correct.view(torch.int16), wrong.view(torch.int16))


def test_after_fc2_is_opt_in_and_routes_to_owner_only_when_enabled(monkeypatch):
    routed = ExpertOutputs(torch.ones(2, 8), "permuted_partial",
                           RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None),
                           reduce=lambda rows: rows, weight_after_fc2=True)
    monkeypatch.setattr(OC, "_engine_reference", lambda *args: "reference")
    monkeypatch.setattr(OC, "_engine_owner_dispatch", lambda *args: "candidate")
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda *args: None)
    combine = OC.engine_combine_owner(lambda *args: None)
    monkeypatch.delenv("ISOEXEC_MOE_AFTER_FC2_OWNER", raising=False)
    assert combine(routed, None, owner_group=object()) == "reference"
    monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", "1")
    assert combine(routed, None, owner_group=object()) == "candidate"
    with pytest.raises(ValueError, match="FP32 routed wire"):
        OC._unified_push_src(8, 8, True, True, True, True, True)


@pytest.mark.parametrize("world,topk", [(2, 1), (8, 8)])
def test_after_fc2_sm90_aot_keeps_rounding_and_no_fma(world, topk):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = OC._unified_owner_push_kernel(world, topk, False, True, False, True, True)
    signature, constants = {}, {"K": topk, "BLOCK": 1024}
    for name in fn.arg_names:
        if name.startswith(("rin", "sin")) or name == "route_probs_ptr":
            signature[name] = "*fp32"
        elif name == "gate_ptr" or name.startswith("out") and name != "out_base":
            signature[name] = "*bf16"
        elif name == "rows_ptr":
            signature[name] = "*i32"
        elif name in constants:
            signature[name] = "constexpr"
        else:
            signature[name] = "i32"
    assembly = triton.compile(ASTSource(fn, signature, constexprs=constants),
                              target=GPUTarget("cuda", 90, 32), options={"enable_fp_fusion": False}).asm
    assert assembly["ttir"].count("arith.truncf") >= 6
    assert not any(op.startswith("fma") for op in _fp_ops(assembly["ptx"]))


@pytest.mark.parametrize("case", ["peer_ineligible", "uncached_capture", "wrong_candidate", "accepted", "cached_refusal", "warm_capture", "capture_pool_missing"])
def test_admission_group_gate_capture_and_semantic_cache(monkeypatch, case):
    import sys
    from types import ModuleType

    routed = ExpertOutputs(torch.ones(6, 8), "permuted_partial",
                           RoutingResult(torch.tensor([[3, 1]] * 3), torch.full((3, 2), 0.5), None),
                           extra=torch.zeros(6, dtype=torch.int64), reduce=lambda rows: rows,
                           combine_rows=torch.arange(6, dtype=torch.int32).reshape(3, 2), weight_after_fc2=True)
    reference = torch.ones(3, 8)
    votes, exchanges = [], []
    backend = ModuleType("pik.allreduce")

    def agree(value, *args):
        votes.append(value)
        return False if case == "peer_ineligible" else value

    backend._agree = agree
    backend._capturing = lambda: case in ("uncached_capture", "warm_capture", "capture_pool_missing")
    monkeypatch.setitem(sys.modules, "pik.allreduce", backend)
    monkeypatch.setattr(OC, "_ensure_canonical_pik", lambda: None)
    monkeypatch.setattr(OC.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(OC.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(OC, "_engine_eligible", lambda *args: (True, ""))
    monkeypatch.setattr(OC, "_engine_reference", lambda *args: reference)
    monkeypatch.setattr(OC, "_build_rows", lambda extra, tokens, topk, rows: rows)
    monkeypatch.setattr(OC, "_wire_bf16", lambda *args: pytest.fail("After-FC2 must keep FP32 wire"))
    admissions = {(2, 3, 8, 2, "fp32", None, False, False): True}
    key = (2, 3, 8, 2, "fp32", None, False, True)
    if case == "cached_refusal":
        admissions[key] = False
    elif case in ("warm_capture", "capture_pool_missing"):
        admissions[key] = True
    monkeypatch.setattr(OC, "_unified_pools_ready", lambda *args: case != "capture_pool_missing")
    monkeypatch.setattr(OC, "_ENGINE_ADMIT", admissions)
    monkeypatch.setattr(OC, "_ENGINE_COUNTS", dict.fromkeys(OC._ENGINE_COUNTS, 0))
    monkeypatch.setattr(OC, "_WIRE_GROUP", {})

    class Exchange:
        def __init__(self, *args, route_probabilities):
            assert votes == ([] if case == "warm_capture" else [True]), "Candidate admission must be agreed"
            assert args[-1] is False
            assert torch.equal(route_probabilities, OC._owner_route_probabilities(routed))
            exchanges.append(True)

        def run(self):
            return reference.bfloat16() + (1 if case == "wrong_candidate" else 0)

    monkeypatch.setattr(OC, "_UnifiedExchange", Exchange)
    actual = OC._engine_owner_dispatch(routed, None, object(), lambda *args: None)
    assert torch.equal(actual, reference)
    if case == "accepted":
        assert votes == [True, True] and exchanges == [True] and admissions[key] is True
    elif case == "warm_capture":
        assert not votes and exchanges == [True] and admissions[key] is True
    elif case == "wrong_candidate":
        assert votes == [True, False] and exchanges == [True] and admissions[key] is False
    elif case == "peer_ineligible":
        assert votes == [True, False] and not exchanges and admissions[key] is False
    else:
        assert not votes and not exchanges
        if case == "uncached_capture":
            assert key not in admissions  # old semantic cache entry must not admit this variant


@pytest.mark.parametrize("peer_mode", [(False, True), (True, False), (False, False), (True, True)])
def test_rank_mode_negotiation_refuses_disagreement_before_dispatch(monkeypatch, peer_mode):
    import sys
    from types import ModuleType

    backend = ModuleType("pik.allreduce")
    backend._capturing = lambda: False
    monkeypatch.setitem(sys.modules, "pik.allreduce", backend)
    monkeypatch.setattr(OC, "_ensure_canonical_pik", lambda: None)
    monkeypatch.setattr(OC.dist, "get_world_size", lambda group: 2)

    def gather(peers, local, group):
        peers[:2].copy_(local)
        peers[2:].copy_(torch.tensor(peer_mode, dtype=torch.int32))

    monkeypatch.setattr(OC.dist, "all_gather_into_tensor", gather)
    if peer_mode == (True, True):
        OC._negotiate_owner_mode((True, True), object(), torch.device("cpu"))
    else:
        with pytest.raises(RuntimeError, match="ranks disagree"):
            OC._negotiate_owner_mode((True, True), object(), torch.device("cpu"))
    backend._capturing = lambda: True
    with pytest.raises(RuntimeError, match="eager warmup"):
        OC._negotiate_owner_mode((True, True), object(), torch.device("cpu"))


def test_mode_negotiation_is_cached_and_later_mutation_refused(monkeypatch):
    routed = ExpertOutputs(torch.ones(2, 8), "permuted_partial",
                           RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None),
                           reduce=lambda rows: rows, weight_after_fc2=True)
    calls = []
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda *args: calls.append(args[0]))
    monkeypatch.setattr(OC, "_engine_reference", lambda *args: "reference")
    monkeypatch.delenv("ISOEXEC_MOE_AFTER_FC2_OWNER", raising=False)
    combine, group = OC.engine_combine_owner(lambda *args: None), object()
    assert combine(routed, None, owner_group=group) == "reference"
    assert combine(routed, None, owner_group=group) == "reference"
    assert calls == [(False, True)]
    monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", "1")
    with pytest.raises(RuntimeError, match="mode changed"):
        combine(routed, None, owner_group=group)


@pytest.mark.parametrize("invalid", ["strided_rows", "fp16_probabilities", "wrong_probability_shape", "float_indices", "missing_rows"])
def test_after_fc2_eligibility_refuses_unsupported_layout(monkeypatch, invalid):
    routing = RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None)
    routed = ExpertOutputs(torch.ones(2, 8), "permuted_partial", routing,
                           extra=torch.zeros(2, dtype=torch.int64), reduce=lambda rows: rows,
                           combine_rows=torch.tensor([[1, 0]], dtype=torch.int32), weight_after_fc2=True)
    if invalid == "strided_rows":
        routed.rows = torch.ones(8, 2).t()
    elif invalid == "fp16_probabilities":
        routed.routing = RoutingResult(routing.indices, routing.probs.half(), None)
    elif invalid == "wrong_probability_shape":
        routed.routing = RoutingResult(routing.indices, routing.probs[:, :1], None)
    elif invalid == "float_indices":
        routed.routing = RoutingResult(routing.indices.float(), routing.probs, None)
    else:
        routed.combine_rows = None
    monkeypatch.setattr(OC, "owner_combine_enabled", lambda: True)
    monkeypatch.setattr(OC, "_engine_p2p", lambda group: True)
    monkeypatch.setattr(OC.dist, "get_world_size", lambda group: 2)
    okay, reason = OC._engine_eligible(routed, None, object(), 1, 2)
    assert not okay and "After-FC2 owner" in reason


def test_noncontiguous_probabilities_are_mapped_to_contiguous_expert_order():
    values = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2]
    indices = torch.tensor([[8, 1, 4], [3, 9, 1]])
    routed = ExpertOutputs(torch.empty(6, 8), "permuted_partial", RoutingResult(indices, values, None),
                           weight_after_fc2=True)
    actual = OC._owner_route_probabilities(routed)
    assert actual.is_contiguous()
    assert torch.equal(actual, torch.tensor([[2., 4., 0.], [10., 6., 8.]]))


def test_each_binding_factory_owns_its_mode_cache(monkeypatch):
    from isoexec.ops.moe.moe_sm90.integration import engine_combine

    calls = []
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda mode, *args: calls.append(mode))
    monkeypatch.setattr(OC, "_engine_owner_dispatch", lambda *args: "candidate")
    monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", "1")
    # resolve(moe.combine) calls this factory for each OpTable.bind(site=...).
    first, second = engine_combine(lambda *args: None), engine_combine(lambda *args: None)
    assert first is not second
    group = object()
    routed = ExpertOutputs(torch.ones(2, 8), "permuted_partial",
                           RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None),
                           reduce=lambda rows: rows)
    for _ in range(2):
        routed.weight_after_fc2 = True
        assert first(routed, None, owner_group=group) == "candidate"
        routed.weight_after_fc2 = False
        assert second(routed, None, owner_group=group) == "candidate"
    assert calls == [(True, True), (True, False)]
