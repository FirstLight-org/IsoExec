"""Run-to-run determinism of the vendored GDN and conv kernels on production per-rank geometry.

Each kernel is called twice on identical inputs with the pinned autotune configs, at the 35B-A3B
engine/trainer/oracle per-rank geometries and on chunk-boundary packs; the two results must agree
in value AND raw bit pattern, and be finite (identical NaN bits can hide divergent, unnormalized
delta-rule inputs). The vendored copies were qualified when they went in and are not compared
with an installed upstream.

Run: one GPU, as a file.
    python -m isoexec.ops.gdn.shared.tests.gdn_vendored_determinism_gpu
"""

from __future__ import annotations

import argparse

import torch

# The vendored tree only: every comparison below is one of its kernels against itself.
from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as V_CONV
from isoexec.ops.gdn.shared import vendored_fla as _VF  # noqa: F401  (package marker)
from isoexec.ops.gdn.shared.gdn_batch_invariant import pin_fla_autotune_configs
from isoexec.ops.gdn.shared.vendored_fla import chunk_delta_h as V_cdh
from isoexec.ops.gdn.shared.vendored_fla import chunk_scaled_dot_kkt as V_kkt
from isoexec.ops.gdn.shared.vendored_fla import cumsum as V_cumsum
from isoexec.ops.gdn.shared.vendored_fla import fused_recurrent as V_rec
from isoexec.ops.gdn.shared.vendored_fla import fused_sigmoid_gating as V_gate
from isoexec.ops.gdn.shared.vendored_fla import index as V_index
from isoexec.ops.gdn.shared.vendored_fla import l2norm as V_l2
from isoexec.ops.gdn.shared.vendored_fla import solve_tril as V_tril
from isoexec.ops.gdn.shared.vendored_fla import wy_fast as V_wy

DEV = "cuda"
BF16, FP32 = torch.bfloat16, torch.float32
C = 64  # FLA_CHUNK_SIZE, and the CPR chunk size the contract pins

# Real per-rank GDN geometry. (label, H key heads, HV value heads, K, V, conv width).
# Qwen3.5-35B-A3B: linear_num_key_heads=16, linear_num_value_heads=32, linear_{key,value}_head_dim=128,
# linear_conv_kernel_dim=4 -- see models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json.
# tiny_qwen35_moe: 4 / 8 / 32 / 32 / 4.
GEOMETRIES = [
    ("35B-A3B @TP8 (engine)", 2, 4, 128, 128, 4),
    ("35B-A3B @TP4 (trainer)", 4, 8, 128, 128, 4),
    ("35B-A3B @TP1 (oracle)", 16, 32, 128, 128, 4),
    ("tiny_qwen35_moe @TP1", 4, 8, 32, 32, 4),
]

#: Packed batches. Chosen for the branch structure of the chunk kernels, not for coverage volume:
#: 56/120 are gate_vllm_engine's chunk-boundary prompts (their decodes cross 64 and 128), 64/128 are
#: exact chunk multiples, 1 is decode, and the ragged pack mixes all of it in one varlen launch.
PACKS = [
    ("decode x4", [1, 1, 1, 1]),
    ("boundary 56/120 (gate_vllm_engine --extra-lengths)", [56, 120]),
    ("exact chunks 64/128", [64, 128]),
    ("boundary crossings 57/63/65/127/129", [57, 63, 65, 127, 129]),
    ("ragged pack", [150, 37, 64, 201, 1]),
]


class Ledger:
    """Record value differences, raw-bit differences and non-finite outputs per kernel/case.
    Non-finite output fails: identical NaN bits can hide divergent, unnormalized delta-rule inputs."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, int, int, int]] = []

    def add(self, kernel: str, case: str, ours, theirs) -> None:  # theirs = the second call
        """One comparison row. Records differing elements by value AND by bit pattern."""
        ours_l = ours if isinstance(ours, (tuple, list)) else (ours,)
        theirs_l = theirs if isinstance(theirs, (tuple, list)) else (theirs,)
        assert len(ours_l) == len(theirs_l)
        ne = bits = nonfinite = 0
        for a, b in zip(ours_l, theirs_l):
            if a is None and b is None:
                continue
            if (a is None) != (b is None) or a.shape != b.shape or a.dtype != b.dtype:
                ne = bits = -1
                break
            if a.dtype.is_floating_point:
                nonfinite += int((~torch.isfinite(a)).sum()) + int((~torch.isfinite(b)).sum())
            if not torch.equal(a, b):
                ne += int((a != b).sum())
            bits += int(_bitdiff(a, b))
        self.rows.append((kernel, case, ne, bits, nonfinite))

    @property
    def ok(self) -> bool:
        return all(ne == 0 and bits == 0 and nf == 0 for _, _, ne, bits, nf in self.rows)

    def report(self) -> str:
        out = []
        for kernel, case, ne, bits, nf in self.rows:
            status = "PASS" if ne == 0 and bits == 0 and nf == 0 else "FAIL"
            out.append(f"  [{status}] {kernel:<38} {case:<62} ne={ne} bitdiff={bits} nonfinite={nf}")
        return "\n".join(out)


def _bitdiff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing elements by raw bit pattern. Catches -0.0 vs +0.0, which torch.equal cannot."""
    if a.dtype.is_floating_point:
        width = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
        it = width.get(a.dtype, torch.int64)
        return int((a.contiguous().view(it) != b.contiguous().view(it)).sum())
    return int((a != b).sum())


def _cu(lens):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)


def _inputs(lens, H, HV, K, V, seed):
    """One packed GDN batch in the kernels' calling convention (B=1 varlen)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(lens)
    return dict(
        q_raw=torch.randn(1, T, H, K, generator=g, device=DEV, dtype=BF16),
        k_raw=torch.randn(1, T, H, K, generator=g, device=DEV, dtype=BF16),
        v=torch.randn(1, T, HV, V, generator=g, device=DEV, dtype=BF16),
        # g is a log-space decay: strictly negative, as -exp(A_log)*softplus(...) always is.
        gl=-torch.nn.functional.softplus(torch.randn(1, T, HV, generator=g, device=DEV, dtype=FP32)),
        beta=torch.rand(1, T, HV, generator=g, device=DEV, dtype=BF16).sigmoid(),
        a=torch.randn(1, T, HV, generator=g, device=DEV, dtype=BF16),
        b=torch.randn(1, T, HV, generator=g, device=DEV, dtype=BF16),
        A_log=torch.randn(HV, generator=g, device=DEV, dtype=FP32),
        dt_bias=torch.randn(HV, generator=g, device=DEV, dtype=FP32),
    )


def _expand(t, H, HV):
    """GQA-expand a [B, T, H, D] tensor to HV heads, the way the chunk kernel requires."""
    if H == HV:
        return t
    return t.repeat_interleave(HV // H, dim=2).contiguous()


def _second_run():
    """The second run's modules: our own vendored copies, called again on the same inputs.

    Every ``_cmp_*`` below calls its kernel once through ``V_*`` and once through this table, so each
    ledger row is a run-to-run comparison of the vendored kernel with itself -- bit pattern, value
    and finiteness -- at production geometry and chunk-boundary packs. Nothing here imports vLLM.
    """
    return dict(
        cdh=V_cdh,
        kkt=V_kkt,
        cumsum=V_cumsum,
        rec=V_rec,
        gate=V_gate,
        index=V_index,
        l2=V_l2,
        tril=V_tril,
        wy=V_wy,
        conv=V_CONV,
    )


# The per-kernel comparisons. Each takes the two module namespaces (here the same tree twice).


def _cmp_l2norm(led, U, t, label):
    """`gdn.l2norm`. l2norm_fwd is called on q and k with the shapes above."""
    for name in ("q_raw", "k_raw"):
        led.add("l2norm_fwd", f"{label} {name}", V_l2.l2norm_fwd(t[name]), U["l2"].l2norm_fwd(t[name]))


def _cmp_index(led, U, cu, label):
    """The chunk index maps. Integer tensors, so any difference is a hard divergence.

    Both sides are `@tensor_cache`d on tensor IDENTITY and the two trees hold separate caches, so
    handing the same `cu` to both is safe -- but each gets a fresh clone anyway, because a stale
    entry here would silently make the comparison compare nothing.
    """
    cu_a, cu_b = cu.clone(), cu.clone()
    led.add(
        "prepare_chunk_indices",
        label,
        V_index.prepare_chunk_indices(cu_a, C),
        U["index"].prepare_chunk_indices(cu_b, C),
    )
    led.add(
        "prepare_chunk_offsets",
        label,
        V_index.prepare_chunk_offsets(cu_a, C),
        U["index"].prepare_chunk_offsets(cu_b, C),
    )


def _cmp_cpr_chain(led, U, t, cu, H, HV, K, V, label):
    """The CPR matched-prep chain, stage by stage -- `gdn_cpr.chunk_boundary_states` verbatim.

    Compared per stage rather than end to end: an end-to-end match can hide two cancelling stage
    divergences, and a stage-level failure names the file to re-vendor.
    """
    # `native_matched_prep` L2-normalises k before this chain: un-normalised inputs can diverge into
    # NaN state, and identical NaN bits would be vacuous agreement.
    k = V_l2.l2norm_fwd(_expand(t["k_raw"], H, HV))
    cu_v, cu_u = cu.clone(), cu.clone()
    ci_v = V_index.prepare_chunk_indices(cu_v, C)
    co_v = V_index.prepare_chunk_offsets(cu_v, C)
    ci_u = U["index"].prepare_chunk_indices(cu_u, C)
    co_u = U["index"].prepare_chunk_offsets(cu_u, C)

    g_v = V_cumsum.chunk_local_cumsum(t["gl"], chunk_size=C, cu_seqlens=cu_v, chunk_indices=ci_v)
    g_u = U["cumsum"].chunk_local_cumsum(t["gl"], chunk_size=C, cu_seqlens=cu_u, chunk_indices=ci_u)
    led.add("chunk_local_cumsum", label, g_v, g_u)

    A_v = V_kkt.chunk_scaled_dot_kkt_fwd(
        k=k, beta=t["beta"], g=g_v, cu_seqlens=cu_v, chunk_indices=ci_v, chunk_size=C, output_dtype=FP32
    )
    A_u = U["kkt"].chunk_scaled_dot_kkt_fwd(
        k=k, beta=t["beta"], g=g_u, cu_seqlens=cu_u, chunk_indices=ci_u, chunk_size=C, output_dtype=FP32
    )
    led.add("chunk_scaled_dot_kkt_fwd", label, A_v, A_u)

    Ai_v = V_tril.solve_tril(A=A_v, cu_seqlens=cu_v, chunk_indices=ci_v, output_dtype=k.dtype)
    Ai_u = U["tril"].solve_tril(A=A_u, cu_seqlens=cu_u, chunk_indices=ci_u, output_dtype=k.dtype)
    led.add("solve_tril", label, Ai_v, Ai_u)

    w_v, u_v = V_wy.recompute_w_u_fwd(
        k=k, v=t["v"], beta=t["beta"], A=Ai_v, g_cumsum=g_v, cu_seqlens=cu_v, chunk_indices=ci_v
    )
    w_u, u_u = U["wy"].recompute_w_u_fwd(
        k=k, v=t["v"], beta=t["beta"], A=Ai_u, g_cumsum=g_u, cu_seqlens=cu_u, chunk_indices=ci_u
    )
    led.add("recompute_w_u_fwd", label, (w_v, u_v), (w_u, u_u))

    h_v, vn_v, fs_v = V_cdh.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w_v,
        u=u_v,
        g=g_v,
        initial_state=None,
        output_final_state=True,
        chunk_size=C,
        cu_seqlens=cu_v,
        chunk_indices=ci_v,
        chunk_offsets=co_v,
    )
    h_u, vn_u, fs_u = U["cdh"].chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w_u,
        u=u_u,
        g=g_u,
        initial_state=None,
        output_final_state=True,
        chunk_size=C,
        cu_seqlens=cu_u,
        chunk_indices=ci_u,
        chunk_offsets=co_u,
    )
    led.add("chunk_gated_delta_rule_fwd_h", f"{label} h/v_new/final", (h_v, vn_v, fs_v), (h_u, vn_u, fs_u))


def _state_pool(lens, HV, V, K):
    """The recurrent kernels' state pool and index map: row 0 is NULL, sequence n owns row n+1."""
    N, Tmax = len(lens), max(lens)
    pool = torch.zeros(N + 1, HV, V, K, dtype=FP32, device=DEV)
    pad = max(64, 1 << (Tmax - 1).bit_length())
    idx = torch.zeros(N, pad, dtype=torch.int32, device=DEV)
    idx[:, 0] = torch.arange(1, N + 1, dtype=torch.int32, device=DEV)
    return pool, idx[:, :Tmax].contiguous()


def _cmp_recurrent(led, U, t, cu, lens, H, HV, K, V, label):
    """`gdn.core` under `ISOEXEC_GDN_KERNEL=recurrent`: the fused recurrent scan.

    `inplace_final_state=True` means the pool is written in place, so each side gets its own.
    """
    q = V_l2.l2norm_fwd(_expand(t["q_raw"], H, HV))
    k = V_l2.l2norm_fwd(_expand(t["k_raw"], H, HV))
    pool_v, idx = _state_pool(lens, HV, V, K)
    pool_u = pool_v.clone()

    def call(mod, pool):
        return mod.fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=t["v"],
            g=t["gl"],
            beta=t["beta"],
            scale=None,
            initial_state=pool,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=False,
        )

    o_v, _ = call(V_rec, pool_v)
    o_u, _ = call(U["rec"], pool_u)
    led.add("fused_recurrent_gated_delta_rule", f"{label} o", o_v, o_u)
    led.add("fused_recurrent_gated_delta_rule", f"{label} state (in place)", pool_v, pool_u)


def _cmp_gating(led, U, t, cu, lens, H, HV, K, V, label):
    """`gdn.gating` + the native core: `fused_sigmoid_gating_delta_rule_update`.

    Called exactly as `gdn_ops.gdn_native_core_kernel` does, including
    `use_qk_l2norm_in_kernel=True` -- the in-kernel l2norm is part of this kernel's schedule.
    """
    q = _expand(t["q_raw"], H, HV)
    k = _expand(t["k_raw"], H, HV)
    pool_v, idx = _state_pool(lens, HV, V, K)
    pool_u = pool_v.clone()

    def call(mod, pool):
        return mod.fused_sigmoid_gating_delta_rule_update(
            A_log=t["A_log"],
            a=t["a"],
            b=t["b"],
            dt_bias=t["dt_bias"],
            q=q,
            k=k,
            v=t["v"],
            initial_state=pool,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
        )

    o_v, _ = call(V_gate, pool_v)
    o_u, _ = call(U["gate"], pool_u)
    led.add("fused_sigmoid_gating_delta_rule_update", f"{label} o", o_v, o_u)
    led.add("fused_sigmoid_gating_delta_rule_update", f"{label} state (in place)", pool_v, pool_u)


def _cmp_conv(led, U, cu, lens, H, HV, K, V, W, label, seed):
    """`conv.short`, both kernels. The varlen prefill launch and the in-place decode window slide.

    D is the mixed_qkv width the model actually convolves: 2*H*K + HV*V per rank.
    """
    D = 2 * H * K + HV * V
    T, N = sum(lens), len(lens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, D, generator=g, device=DEV, dtype=BF16)
    weight = torch.randn(D, W, generator=g, device=DEV, dtype=BF16)
    bias = torch.randn(D, generator=g, device=DEV, dtype=BF16)
    idx = torch.arange(1, N + 1, dtype=torch.int32, device=DEV)
    has0 = torch.zeros(N, dtype=torch.bool, device=DEV)

    def call_fn(mod):
        states = torch.zeros(N + 1, D, W - 1, dtype=BF16, device=DEV)
        y = mod.causal_conv1d_fn(
            x.transpose(0, 1),
            weight,
            bias,
            conv_states=states,
            query_start_loc=cu,
            cache_indices=idx,
            has_initial_state=has0,
            activation="silu",
        ).transpose(0, 1)
        return y.contiguous(), states

    y_v, st_v = call_fn(V_CONV)
    y_u, st_u = call_fn(U["conv"])
    led.add("causal_conv1d_fn", f"{label} y", y_v, y_u)
    led.add("causal_conv1d_fn", f"{label} conv_states", st_v, st_u)

    # Decode: one token per sequence against a populated window, written back in place.
    xd = torch.randn(N, D, generator=g, device=DEV, dtype=BF16)
    base = torch.randn(N + 1, D, W - 1, generator=g, device=DEV, dtype=BF16)
    sidx = torch.arange(1, N + 1, dtype=torch.int32, device=DEV)

    def call_update(mod):
        state = base.clone()
        out = mod.causal_conv1d_update(
            xd.clone(), state, weight, bias, activation="silu", conv_state_indices=sidx
        )
        return out, state

    o_v, s_v = call_update(V_CONV)
    o_u, s_u = call_update(U["conv"])
    led.add("causal_conv1d_update", f"{label} y", o_v, o_u)
    led.add("causal_conv1d_update", f"{label} conv_state (in place)", s_v, s_u)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="one geometry, one pack (smoke)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    U = _second_run()

    n = pin_fla_autotune_configs()
    print(f"[gdn-vendored-determinism] pinned {n} vendored autotuners")

    geometries = GEOMETRIES[:1] if args.quick else GEOMETRIES
    packs = PACKS[1:2] if args.quick else PACKS

    led = Ledger()
    for gi, (glabel, H, HV, K, V, W) in enumerate(geometries):
        for pi, (plabel, lens) in enumerate(packs):
            label = f"{glabel} / {plabel}"
            cu = _cu(lens)
            t = _inputs(lens, H, HV, K, V, seed=1000 * gi + pi)
            _cmp_l2norm(led, U, t, label)
            _cmp_index(led, U, cu, label)
            _cmp_cpr_chain(led, U, t, cu, H, HV, K, V, label)
            _cmp_recurrent(led, U, t, cu, lens, H, HV, K, V, label)
            _cmp_gating(led, U, t, cu, lens, H, HV, K, V, label)
            _cmp_conv(led, U, cu, lens, H, HV, K, V, W, label, seed=1000 * gi + pi)

    print(led.report())
    total = len(led.rows)
    bad = sum(1 for _, _, ne, bits, nf in led.rows if ne or bits or nf)
    print(f"\n[gdn-vendored-determinism] {total - bad}/{total} repeated calls bit-identical (torch.equal AND bit pattern)")
    print("RESULT:", "PASS" if led.ok else "FAIL")
    return 0 if led.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
