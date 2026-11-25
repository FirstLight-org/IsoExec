"""CPU/AOT checks for the unified owner composition: routed fp32 sum seeded by its first term,
shared = bf16(bf16(tree(shared leaves)) * gate_bf16), then bf16(routed + fp32(shared)).
The tests also reject incomplete deferred-reduction wiring and preserve the bridge's shipped source."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

pytest.importorskip("triton")

_real_capability = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda _device=None: (9, 0)
try:
    OC = importlib.import_module("isoexec.ops.moe.moe_sm90.moe_pik_combine_owner")
finally:
    torch.cuda.get_device_capability = _real_capability

from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult, SharedExpertPartial  # noqa: E402


def _compile(world, k, routed_bf16, shared, shared_bf16, gated):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = OC._unified_owner_push_kernel(world, k, routed_bf16, shared, shared_bf16, gated)
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    sig, consts = {}, {}
    for name in fn.arg_names:
        if name.startswith("rin"):
            sig[name] = "*bf16" if routed_bf16 else "*fp32"
        elif name.startswith("sin"):
            sig[name] = "*bf16" if shared_bf16 else "*fp32"
        elif name == "gate_ptr" or (name.startswith("out") and name != "out_base"):
            sig[name] = "*bf16"
        elif name == "rows_ptr":
            sig[name] = "*i32"
        elif name in ("K", "BLOCK"):
            sig[name] = "constexpr"
            consts[name] = {"K": k, "BLOCK": 1024}[name]
        else:
            sig[name] = "i32"
    return triton.compile(
        ASTSource(fn=fn, signature=sig, constexprs=consts),
        target=GPUTarget("cuda", 90, 32),
        options={"enable_fp_fusion": False},
    ).asm


def _fp_ops(ptx: str) -> list[str]:
    ops = []
    for line in ptx.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        op = fields[1] if fields[0].startswith("@") and len(fields) > 1 else fields[0]
        if op.split(".")[0] in ("add", "mul", "fma", "sub", "cvt") and (".f32" in op or ".bf16" in op or ".f16" in op):
            ops.append(op.rstrip(";"))
    return ops


def test_generated_source_pins_the_unified_rounding_schedule():
    src = OC._unified_push_src(8, 8, True, True, True, True)
    # seeded k-sum: the first term is a copy, never `0.0 + x`
    seed = src.index("if j == 0:\n            routed_acc = rt0")
    add = src.index("routed_acc = routed_acc + rt0")
    assert seed < add
    # the routed sum is NEVER rounded before the final add (the bridge core's `routed_bf` is absent)
    assert "routed_bf =" not in src and "routed_acc.to(tl.bfloat16)" not in src
    shared_round = src.index("shared_root_bf = st0.to(tl.bfloat16)")
    gate = src.index("gate_bf = tl.load(gate_ptr + src_t).to(tl.bfloat16)")
    product = src.index("shared_bf = (shared_root_bf.to(tl.float32) * gate_bf.to(tl.float32)).to(tl.bfloat16)")
    final = src.index("result = (routed_acc + shared_bf.to(tl.float32)).to(tl.bfloat16)")
    store = src.index("tl.store(out0 + goff, result")
    assert add < shared_round < gate < product < final < store
    # every bf16 narrowing in the program, by line: shared root, gate (a bf16->bf16 identity cast on
    # the load), gate product, the final sum -- and the routed sum appears in exactly the last one
    narrowing = sorted(line.strip() for line in src.splitlines() if ".to(tl.bfloat16)" in line)
    assert narrowing == sorted(
        [
            "shared_root_bf = st0.to(tl.bfloat16)",
            "gate_bf = tl.load(gate_ptr + src_t).to(tl.bfloat16)",
            "shared_bf = (shared_root_bf.to(tl.float32) * gate_bf.to(tl.float32)).to(tl.bfloat16)",
            "result = (routed_acc + shared_bf.to(tl.float32)).to(tl.bfloat16)",
        ]
    )
    # padded owner slots read token T-1 of the shared peers, never past the buffer
    assert "src_t = tl.minimum(global_t, T - 1)" in src


def test_variants_without_gate_and_without_shared_expert():
    ungated = OC._unified_push_src(4, 2, False, True, False, False)
    assert "gate_ptr" not in ungated and "shared_bf = shared_root_bf" in ungated
    assert "result = (routed_acc + shared_bf.to(tl.float32)).to(tl.bfloat16)" in ungated
    noshared = OC._unified_push_src(4, 2, False, False, False, False)
    assert "sin0" not in noshared and "gate_ptr" not in noshared
    assert "result = routed_acc.to(tl.bfloat16)" in noshared
    assert noshared.count(".to(tl.bfloat16)") == 1  # the residual seam's one round, and nothing else
    # fp32 wire: leaves load without a promotion cast; bf16 wire: every leaf is promoted
    assert ".to(tl.float32)" not in OC._named_tree_src(4, False, inp="rin", tmp="rt", base="rbase", indent="")
    assert OC._named_tree_src(4, True, inp="rin", tmp="rt", base="rbase", indent="").count(".to(tl.float32)") == 4


def test_both_trees_are_the_canonical_balanced_tree_and_the_routed_owner_source_did_not_move():
    src = OC._unified_owner_core(8, 8, True, True, True, True)
    routed = OC._named_tree_src(8, True, inp="rin", tmp="rt", base="rbase", indent="        ")
    shared = OC._named_tree_src(8, True, inp="sin", tmp="st", base="sbase", indent="    ")
    assert routed in src and shared in src
    assert OC._oc_core(8, 8, True) in OC._push_src(8, 8, True)
    assert "if j == 0" not in OC._oc_core(8, 8, True)  # the bridge core seeds with zeros; ours does not


@pytest.mark.parametrize("world,k", [(2, 1), (2, 8), (8, 1), (8, 8)])
def test_sm90_machine_code_has_three_rounds_and_no_fma(world, k):
    """TTIR pins the truncations (the source's three narrowings; the gate's bf16->bf16 cast is not
    one). PTX may fold a `(bf16 + bf16).to(bf16)` / `(bf16 * bf16).to(bf16)` into a native bf16
    add/mul -- bit-identical, since two 8-bit significands sum or multiply exactly inside fp32 -- so
    cvt.rn alone is not counted; what must never appear is an FMA (a deleted rounding point)."""
    asm = _compile(world, k, True, True, True, True)
    assert asm["ttir"].count("arith.truncf") == 3
    ops = _fp_ops(asm["ptx"])
    assert not any(op.startswith("fma") for op in ops), ops
    # the routed sum reaches the final add as fp32: at least one fp32 add per thread element for it
    assert any(op.startswith("add.f32") or op.startswith("add.rn.f32") for op in ops), ops
    # and the shared branch is a separate family: ungated -> 2 truncations, no shared -> 1
    assert _compile(world, k, False, True, False, False)["ttir"].count("arith.truncf") == 2
    assert _compile(world, k, False, False, False, False)["ttir"].count("arith.truncf") == 1


def _routed(layout, T=4, k=2, H=8, reduce=None):
    routing = RoutingResult(indices=torch.zeros(T, k, dtype=torch.long), probs=torch.ones(T, k), scores_full=None)
    return ExpertOutputs(
        rows=torch.zeros(T * k, H),
        layout=layout,
        routing=routing,
        extra=torch.zeros(T * k, dtype=torch.long),
        reduce=reduce,
    )


def test_binding_is_the_eager_combine_until_the_placement_wires_the_seam():
    calls = []

    def eager(routed, shared):
        calls.append((routed.layout, type(shared).__name__))
        return torch.zeros(4, 8)

    combine = OC.engine_combine_owner(eager)
    assert "owner_group" in inspect.signature(combine).parameters  # what sharded_op_table keys on
    combine(_routed("permuted"), None)
    combine(_routed("permuted"), torch.zeros(4, 8, dtype=torch.bfloat16))
    assert calls == [("permuted", "NoneType"), ("permuted", "Tensor")]
    # half a seam is refused, never guessed at
    with pytest.raises(RuntimeError, match="half a seam"):
        combine(_routed("permuted_partial", reduce=lambda r: r), None)  # deferred rows, no group
    with pytest.raises(RuntimeError, match="half a seam"):
        combine(_routed("permuted"), None, owner_group=object())  # a group, nothing deferred
    with pytest.raises(RuntimeError, match="half a seam"):
        combine(  # deferred rows + group, but a plain shared tensor (its fc2 tree already ran)
            _routed("permuted_partial", reduce=lambda r: r), torch.zeros(4, 8), owner_group=object()
        )


def test_owner_rows_slice_is_a_view_when_divisible_and_zero_padded_otherwise():
    rows = torch.arange(8 * 2, dtype=torch.int32).view(8, 2)
    mine, s, Tp = OC._engine_my_rows(rows, 8, 2, 4, 1)
    assert (s, Tp) == (2, 8) and mine.data_ptr() == rows[2:4].data_ptr()
    rows = torch.arange(7 * 2, dtype=torch.int32).view(7, 2)
    mine, s, Tp = OC._engine_my_rows(rows, 7, 2, 4, 3)
    assert (s, Tp) == (2, 8) and mine.dtype == torch.int32 and mine.tolist() == [[12, 13], [0, 0]]


def test_shared_partial_eager_is_finish_times_gate():
    part = torch.full((3, 4), 2.0)
    sp = SharedExpertPartial(
        partial=part, gate=torch.full((3, 1), 0.5, dtype=torch.bfloat16), finish=lambda p: p.to(torch.bfloat16)
    )
    assert torch.equal(sp.eager(), torch.full((3, 4), 1.0, dtype=torch.bfloat16))
    assert torch.equal(SharedExpertPartial(part, None, lambda p: p).eager(), part)


@pytest.mark.parametrize('gated', [False, True])
def test_reference_consumes_borrowed_routed_reduction_before_shared_reuses_pool(gated):
    """PIK's large reductions lend the same output pool to both fc2 trees."""
    T, k, H = 3, 4, 8
    original = torch.arange(T*k*H,dtype=torch.float32).reshape(T*k,H)/257
    pool = torch.empty_like(original)
    pointers = []
    def reduce(rows):
        pool.copy_(rows)
        pointers.append(pool.data_ptr())
        return pool
    def finish(partial):
        shared_view = pool[:T]
        shared_view.copy_(partial)
        pointers.append(shared_view.data_ptr())
        return shared_view.to(torch.bfloat16)
    def fold(rows):
        values = rows.reshape(T,k,H)
        out = values[:,0].clone()
        for j in range(1,k):
            out = out + values[:,j]
        return out
    def eager(routed, shared):
        out = fold(routed.rows)
        return out if shared is None else out + shared
    routed = _routed('permuted_partial',T=T,k=k,H=H,reduce=reduce)
    routed.rows = original
    partial = torch.full((T,H),-0.125)
    gate = torch.full((T,1),0.75,dtype=torch.bfloat16) if gated else None
    shared = SharedExpertPartial(partial,gate,finish)
    expected_shared = partial.to(torch.bfloat16)
    if gate is not None:
        expected_shared = expected_shared * gate
    expected = fold(original) + expected_shared
    actual = OC._engine_reference(routed,shared,eager)
    assert pointers[0] == pointers[1], 'fixture must reuse the same borrowed reduction storage'
    assert actual.dtype == torch.float32
    assert torch.equal(actual,expected)


def test_stage_and_eligibility_refuse_when_the_flags_are_off(monkeypatch):
    monkeypatch.setenv("ISOEXEC_MOE_PIK_OWNER_COMBINE", "0")
    assert OC.stage_shared_partial(4, 8, torch.bfloat16, object(), torch.device("cpu")) is None
    ok, why = OC._engine_eligible(_routed("permuted_partial", reduce=lambda r: r), None, None, 4, 2)
    assert not ok and "OWNER_COMBINE" in why
    monkeypatch.setenv("ISOEXEC_MOE_PIK_OWNER_COMBINE", "1")
    monkeypatch.setenv("ISOEXEC_MOE_SHARED_OWNER_FUSION", "0")
    shared = SharedExpertPartial(torch.zeros(4, 8), None, lambda p: p)
    ok, why = OC._engine_eligible(_routed("permuted_partial", reduce=lambda r: r), shared, None, 4, 2)
    assert not ok and "routed-only" in why
