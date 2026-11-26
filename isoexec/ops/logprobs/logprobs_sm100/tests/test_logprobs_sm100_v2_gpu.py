"""logprobs_sm100 **recipe v2** gate battery (SM100 / B200). Bit-pattern comparisons only."""

from __future__ import annotations

import sys

import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.logprobs.logprobs_sm100 import ops, spec
from isoexec.ops.logprobs.logprobs_sm100.tests import test_logprobs_sm100_gpu as v1_battery

DEV = "cuda"
BF = torch.bfloat16
V1, V2 = spec.RECIPE_VERSION, spec.RECIPE_VERSION_V2


def _skip() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0)


class pinned:
    """``ops.RECIPE`` pinned for the duration, restored afterwards."""

    def __init__(self, recipe: int) -> None:
        self.recipe = recipe

    def __enter__(self):
        self.was = ops.RECIPE
        ops.select_recipe(self.recipe)
        return self

    def __exit__(self, *exc):
        ops.select_recipe(self.was)
        return False


def _leaves(x, recipe, leaves_local=8, temperature=None):
    return ops.leaf_states(x, leaves_local, temperature=temperature, recipe=recipe)



V1_RELATIONS = (
    "test_accuracy_vs_float64",
    "test_row_batch_and_view_invariance",
    "test_tp_shard_invariance",
    "test_hazards",
    "test_cuda_graph_and_no_host_sync",
    "test_backward_vs_float64",
    "test_degenerate",
    "test_repeat_launch_and_graph_are_deterministic_across_processes_note",
    "test_round2_sampled_wrapper_is_bit_identical_to_the_round1_wrapper",
)


def test_recipe_v2_passes_every_relation_recipe_v1_is_gated_on():
    for name in V1_RELATIONS:
        assert hasattr(v1_battery, name), name
    if _skip():
        return
    with pinned(V2):
        assert ops.RECIPE == V2
        for name in V1_RELATIONS:
            print(f"  [v2] {name}")
            getattr(v1_battery, name)()



def test_v2_max_is_exactly_v1_max():
    if _skip():
        return
    g = torch.Generator(device=DEV).manual_seed(20)
    for V in (64, 4096, 151680, 154880, 248320):
        for dt in (torch.float32, BF):
            x = (torch.randn(13, V, device=DEV, generator=g) * 5).to(dt)
            m1, s1 = _leaves(x, V1)
            m2, s2 = _leaves(x, V2)
            assert bitwise_equal(m1, m2), (V, dt, "max")
            xs = torch.sort(x.float(), -1).values.to(dt).contiguous()
            assert bitwise_equal(_leaves(xs, V1)[0], _leaves(xs, V2)[0]), (V, dt, "sorted max")
            t = torch.tensor([0.7], device=DEV)
            assert bitwise_equal(_leaves(x, V1, temperature=t)[0], _leaves(x, V2, temperature=t)[0]), (V, dt, "temp")
    haz = _hazard_rows()
    assert bitwise_equal(_leaves(haz, V1)[0], _leaves(haz, V2)[0]), "hazard max"
    print("  v2 max == v1 max, bit for bit: 5 vocabularies x 2 dtypes, sorted rows, temperature, hazards")


def test_v2_and_v1_really_are_different_programs():
    """The version bump is not cosmetic."""
    if _skip():
        return
    g = torch.Generator(device=DEV).manual_seed(21)
    V = 151680
    x = torch.randn(64, V, device=DEV, generator=g) * 5
    s1, s2 = _leaves(x, V1)[1], _leaves(x, V2)[1]
    differing_leaves = int((s1.view(torch.int32) != s2.view(torch.int32)).sum())
    y1, y2 = ops.full_logprobs(x, recipe=V1), ops.full_logprobs(x, recipe=V2)
    assert not bitwise_equal(y1, y2), "v2 must not be a bit-identical rename of v1"
    d = (y1 - y2).abs().max().item()
    assert d < 1e-5, d  # the same function, a different rounding schedule -- not a different answer
    print(f"  v1 != v2: {differing_leaves}/{s1.numel()} leaf sums differ, max |v1 - v2| = {d:.2e} "
          f"(both within 1e-5 of float64, next test)")


def test_v2_merge_is_v1_merge_measured_not_argued():
    if _skip():
        return
    g = torch.Generator(device=DEV).manual_seed(22)
    V = 154880
    x = torch.randn(48, V, device=DEV, generator=g) * 4
    x[0, : V // 8] = float("-inf")  # an absent leaf, so the merge's -inf branch is exercised
    x[1, :] = float("-inf")
    t = torch.randint(0, V, (48,), device=DEV, generator=g).to(torch.int64)
    t[2] = -1
    m, s = _leaves(x, V1)
    out = {}
    for recipe in (V1, V2):
        rM = torch.empty(48, device=DEV, dtype=torch.float32)
        rL = torch.empty_like(rM)
        lse = torch.empty(48, device=DEV, dtype=torch.float64)
        sm = torch.empty_like(rM)
        ops._ext(recipe).merge_rows(m, s, rM, rL, lse, sm, x, t, None, V, None)
        out[recipe] = (rM, rL, lse, sm)
    for a, b, what in zip(out[V1], out[V2], ("M", "L32", "lse", "sampled")):
        assert bitwise_equal(a, b), what
    lse = out[V1][2]
    gy = torch.randn(48, device=DEV, generator=g)
    dxs = []
    for recipe in (V1, V2):
        dx = torch.empty_like(x)
        ops._ext(recipe).grad_rows(dx, x, lse, gy.contiguous(), t.contiguous(), 8)
        dxs.append(dx)
    assert bitwise_equal(dxs[0], dxs[1]), "grad_rows"
    print("  v2's merge_rows/grad_rows on v1's leaf states == v1's, bit for bit (M, L32, lse, sampled, dx)")


def test_accuracy_of_both_recipes_vs_float64():
    if _skip():
        return
    g = torch.Generator(device=DEV).manual_seed(23)
    worst = {V1: 0.0, V2: 0.0}
    for V in (64, 4096, 151680, 154880, 248320):
        x = torch.randn(17, V, device=DEV, generator=g) * 5
        xs = torch.sort(x, -1).values.contiguous()  # rescales v1's normalizer at EVERY column
        for src, label in ((x, "random"), (xs, "ascending")):
            ref = torch.log_softmax(src.double(), -1)
            errs = {}
            for recipe in (V1, V2):
                e = (ops.full_logprobs(src, recipe=recipe).double() - ref).abs().max().item()
                errs[recipe] = e
                worst[recipe] = max(worst[recipe], e)
                assert e < 1e-5, (V, label, recipe, e)
            print(f"  V={V:6d} {label:9s}: v1 {errs[V1]:.2e}  v2 {errs[V2]:.2e}")
    print(f"  worst over all shapes: v1 {worst[V1]:.2e}, v2 {worst[V2]:.2e}")


def test_v2_temperature_and_fused_full_row_agree_with_the_unfused_path():
    if _skip():
        return
    g = torch.Generator(device=DEV).manual_seed(24)
    V = 151680
    for dt in (torch.float32, BF):
        x = (torch.randn(24, V, device=DEV, generator=g) * 3).to(dt)
        t = torch.randint(0, V, (24,), device=DEV, generator=g)
        full = ops.full_logprobs(x, recipe=V2)
        samp = ops.sampled_logprobs(x, t, recipe=V2)
        assert bitwise_equal(full.gather(1, t[:, None]).squeeze(1), samp), dt
        temp = torch.tensor([0.6], device=DEV)
        ref = torch.log_softmax((x.float() / temp).double(), -1)
        err = (ops.full_logprobs(x, temperature=temp, recipe=V2).double() - ref).abs().max().item()
        assert err < 1e-2 if dt is BF else err < 1e-5, (dt, err)
    print("  v2: fused full row == sampled at the target, temperature applied inside the kernel")


def test_signed_zero_max_tie_is_the_one_documented_difference_in_the_max():
    if _skip():
        return
    V = 4096
    x = torch.zeros(3, V, device=DEV)
    x[0, 0::2] = -0.0            # -0 first, then +0 at every other column
    x[0, 1::2] = 0.0
    x[1] = -x[0]                 # the other order
    x[2, :] = -0.0               # every column -0: no tie between the two signs
    m1, _ = _leaves(x, V1)
    m2, _ = _leaves(x, V2)
    signs1 = torch.signbit(m1)
    signs2 = torch.signbit(m2)
    assert_hazard_exercised("signed_zero_max_tie", int((signs1 != signs2).sum()))
    assert not signs2[:2].any(), "v2's fmaxf must return +0 on a +-0 tie"
    assert signs1[2].all() and signs2[2].all(), "an all -0 leaf has no tie: both keep -0"
    y1, y2 = ops.full_logprobs(x, recipe=V1), ops.full_logprobs(x, recipe=V2)
    assert bitwise_equal(y1, y2), "the +-0 max tie must not reach the output"
    print(f"  signed-zero max tie: {int((signs1 != signs2).sum())} leaf maxima differ in the sign of "
          f"zero, 0 output bits differ")


def test_the_row_select_relation_is_recipe_invariant():
    if _skip():
        return
    from isoexec.ops.logprobs.eager import lm_head_slice_identity

    g = torch.Generator(device=DEV).manual_seed(25)
    V = 248320
    x = (torch.randn(64, V, device=DEV, generator=g) * 4).to(BF)
    sel = torch.tensor([0, 3, 17, 40, 63], device=DEV)
    for recipe in (V1, V2):
        with pinned(recipe):
            h = torch.randn(64, 128, device=DEV, generator=g).to(BF)
            assert bitwise_equal(lm_head_slice_identity(h), h), recipe
            full = ops.full_logprobs(x)
            part = ops.full_logprobs(x[sel].contiguous())
            assert bitwise_equal(part, full[sel]), recipe
            t = torch.randint(0, V, (64,), device=DEV, generator=g)
            assert bitwise_equal(ops.sampled_logprobs(x[sel].contiguous(), t[sel]),
                                ops.sampled_logprobs(x, t)[sel]), recipe
    print("  row-select composition holds under BOTH recipes; lm_head_slice is recipe-independent")


def test_the_declared_impl_id_follows_the_pin():
    from isoexec.ops.logprobs.logprobs_sm100 import integration

    with pinned(V1):
        assert integration.IMPL == spec.IMPL == "sm100_online_leaf8_v1"
    with pinned(V2):
        assert integration.IMPL == spec.IMPL_V2 == "sm100_twopass_leaf8_v2"
    assert ops.default_recipe(BF, 151680) == V2 and ops.default_recipe(torch.float32, 151680) == V1
    assert ops.default_recipe(BF, 4096) == V1, "an unmeasured class must fall back to v1"
    print("  integration.IMPL tracks ops.RECIPE; DEFAULT_RECIPE_BY_CLASS falls back to v1 off-table")


def _hazard_rows() -> torch.Tensor:
    """The hazard matrix of v1's ``test_hazards``, verbatim, for the max-equality comparison."""
    V = 4096
    g = torch.Generator(device=DEV).manual_seed(3)
    x = torch.randn(12, V, device=DEV, generator=g)
    x[0, 100:3000] = float("-inf")
    x[1, :] = float("-inf")
    x[2, 7] = float("nan")
    x[3, :] = 2.5
    x[4, 5:9] = 1e-40
    x[4, 9:12] = -1e-40
    x[5, 3] = float("inf")
    x[6, V // 2:] = float("-inf")
    x[7] = torch.linspace(-1e4, 1e4, V, device=DEV)
    x[8, 0] = 3e38
    x[9, :] = -0.0
    x[10, :100] = 1e-45
    return x


def main():
    if _skip():
        print("SKIP: no SM100 (B200) device")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK {len(tests)} v2 tests (recipe pin restored to {ops.RECIPE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
