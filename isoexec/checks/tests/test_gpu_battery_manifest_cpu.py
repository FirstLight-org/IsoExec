"""Index isolated GPU batteries and check manifest completeness in the CPU lane."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRE_ENV = "ISOEXEC_GPU_BATTERIES"


@dataclass(frozen=True)
class Battery:
    """One battery, the way its own docstring says to run it."""

    path: str
    claim: str
    mode: str = "script"
    argv: tuple[str, ...] = ()
    timeout_s: int = 900
    extra_env: dict = field(default_factory=dict)

    def command(self) -> list[str]:
        target = str(_REPO_ROOT / self.path)
        if self.mode == "pytest":
            return [sys.executable, "-m", "pytest", target, "-q"] + list(self.argv)
        return [sys.executable, target] + list(self.argv)


BATTERIES: tuple[Battery, ...] = (
    Battery(
        "isoexec/ops/collectives/tests/context_parallel_seams_gpu.py",
        "CP=2 trainer seams on one SM100 GPU (two Gloo ranks): the gqa_sm100 core's CP form and GDN's "
        "head-wise region give every local row and gradient the CP=1 bits.",
        mode="pytest",
    ),
    Battery(
        "isoexec/runtimes/vllm/tests/weights_sleep_gpu.py",
        "exact level-2 sleep on vLLM's real CuMem pool: every non-Parameter byte of the weights pool survives "
        "sleep(2) + wake_up, and the pool is bitwise identical once Parameters are rewritten (1 GPU).",
    ),
    Battery(
        "isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py",
        "SM100 router selection, canonical combine order, expert segment-tree invariance, and float64 accuracy.",
    ),
    Battery(
        "isoexec/ops/mm/mm_sm100/tests/mm_sm100_battery_gpu.py",
        "SM100 fixed-segment GEMM accuracy and bitwise invariance across execution plans and batch composition.",
    ),
    Battery(
        "isoexec/ops/attention/tests/eager_no_runtime_gpu.py",
        "the eager attention binding runs with no runtime package importable (needs a fresh interpreter: "
        "it installs a meta-path finder).",
    ),
    Battery(
        "isoexec/ops/mm/tests/mm_fwd_only_bmm_gpu.py",
        "the forward-scoped bmm replacement is active in forward and absent in backward.",
    ),
    Battery(
        "isoexec/ops/mm/tests/mm_column_slice_gpu.py",
        "a column slice of the head GEMM is exact, and the provider that actually served is the "
        "one reported.",
        mode="pytest",
    ),
    Battery(
        "isoexec/ops/moe/tests/shared_expert_biteq_gpu.py",
        "moe.shared_expert:eager is row/batch invariant -- a token's output is bit-identical alone, "
        "in a batch of 8 and at position 5000 of 8192 -- and the two mm providers give it the same "
        "bits. That invariance IS the trainer-long-batch vs engine-decode-row claim.",
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/moe/moe_sm90/tests/moe_router_fused_o2_gpu.py",
        "TWIN_PROOFS['moe.route.sorted_topk'] in full: the fused O2 router makes the same selection "
        "and the same probs as the dense reference, under the (score desc, index asc) tie rule.",
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/logprobs/tests/logprobs_head_biteq_gpu.py",
        "logprobs.head:eager is ROW-INVARIANT -- a row's logits are the same bits scored alone as "
        "inside a batch of 8192 -- which is the half of TWIN_PROOFS['logprobs.row_select'] a CPU "
        "cannot decide. Composed with tests/test_row_select_biteq_cpu.py it is the engine-vs-trainer "
        "shared-row claim end to end.",
        argv=("--negative-controls", "--vendor"),
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/gdn/shared/tests/gdn_native_kernel_parity_test_gpu.py",
        "half of TWIN_PROOFS['gdn.cpr']: the fused GDN core matches the unfused chain.",
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/gdn/shared/tests/gdn_vendored_determinism_gpu.py",
        "the vendored GDN and conv kernels are run-to-run deterministic (value, bit pattern, finite) at "
        "production per-rank geometry and chunk-boundary packs.",
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/gdn/shared/tests/gdn_conv_vjp_biteq_gpu.py",
        "the GDN conv VJP is bitwise-equal to the reference backward.",
        timeout_s=1800,
    ),
    Battery(
        "isoexec/ops/collectives/tests/pik_narrow_leaf_battery_gpu.py",
        "the fixed-tree reduction holds at narrow leaf counts, including the shared expert's fc2.",
    ),
    Battery(
        "isoexec/ops/moe/tests/moe_context_parallel_gpu.py",
        "the released MoE layer under a CP=2 cut, at EP1 and with the CP pair as the EP=2 group: every "
        "local row's output, router scores and input gradient equal the CP=1 bits, and backward is "
        "deterministic (two Gloo ranks on one SM90 or SM100 GPU).",
        mode="pytest",
        timeout_s=3600,
    ),
    Battery(
        "isoexec/ops/collectives/tests/pik_blackwell_config_invariance_gpu.py",
        "every Blackwell autotune config of pik's leaf-partial kernel is bitwise-equal to the cuBLASLt "
        "leaf at the shipped models' leaf widths (skips past the arch self-check off sm100).",
    ),
)

NOT_RUN_HERE: dict[str, str] = {
    "isoexec/ops/collectives/tests/cp_contiguous_nccl_gpu.py": (
        "a torchrun script (one NCCL rank per GPU, 2 or 4 GPUs): contiguous CP attention + GDN hand-off over real "
        "point-to-point, bitwise vs CP=1; run with torchrun --nproc-per-node N"
    ),
    "isoexec/ops/moe/tests/ep_matrix_gpu.py": (
        "multi-mode EP battery: one torchrun per NCCL layout (2/4 ranks), 8/16-rank simulations on one GPU, then "
        "a compare pass; driven by a session script (see its docstring), not by one pytest process"
    ),
    "isoexec/runtimes/vllm/tests/cpr_apc_engine_gpu.py": (
        "builds vLLM engines (TP1 and TP2) per case from a GDN checkpoint named by ISOEXEC_TEST_GDN_MODEL_PATH, "
        "which no CPU-lane run provides"
    ),
    "isoexec/runtimes/vllm/tests/cpr_v2_engine_gpu.py": (
        "builds a vLLM engine per case from a GDN checkpoint named by ISOEXEC_TEST_GDN_MODEL_PATH, which "
        "no CPU-lane run provides"
    ),
    "isoexec/ops/moe/moe_sm90/tests/owner_graph_tp8_gpu.py": (
        "8-rank NCCL owner-combine graph and saved-reference check; needs torchrun"
    ),
    "isoexec/ops/collectives/tests/pik_capture_pool_lifetime_gpu.py": (
        "2-rank torch.distributed.run; a single-process pytest worker cannot launch it"
    ),
    "isoexec/ops/collectives/tests/rs_bf16_wire_gpu.py": (
        "multi-GPU NCCL/P2P wire test; needs a reserved multi-device allocation"
    ),
    "isoexec/ops/collectives/tests/seq_p2p_gather_gpu.py": (
        "torchrun on an even number (>= 4) of ranks, and its body runs at import -- it asserts the "
        "world size before any function is called, so it can never be collected, only launched"
    ),
    "isoexec/ops/collectives/tests/collectives_multirank_battery_gpu.py": (
        "spawns --tp NCCL ranks itself and needs that many visible GPUs; one pytest worker owns one"
    ),
    "isoexec/ops/collectives/tests/pik_owner_allreduce_gpu.py": (
        "spawns --world NCCL ranks itself (or --sim on one GPU) and needs that many visible GPUs"
    ),
    "isoexec/ops/collectives/tests/pik_gather_vocab_gpu.py": (
        "spawns --world NCCL ranks itself (the engine's symmetric-memory logits gather vs NCCL all-gather) "
        "and needs that many visible GPUs"
    ),
    "isoexec/ops/collectives/tests/pik_owner_lamport_gpu.py": (
        "multi-rank Lamport protocol + fused add+RMSNorm epilogue battery (--world N spawns N ranks); its "
        "single-GPU half (--sim) needs no peers but shares the script"
    ),
    "isoexec/ops/collectives/tests/pik_overlap_gpu.py": (
        "spawns --world NCCL ranks itself (optionally one visible GPU per rank, --ray-sim) or --sim on one GPU"
    ),
    "isoexec/ops/collectives/tests/pik_p2p_groups_gpu.py": (
        "spawns --world NCCL ranks itself (one GPU each) and needs that many visible GPUs"
    ),
    "isoexec/ops/collectives/tests/pik_workspace_lifetime_gpu.py": (
        "requires --out for its workspace ownership, stream-lifetime and graph-replay receipt"
    ),
    "isoexec/runtimes/megatron/tests/grad_kernels_gpu.py": (
        "needs a live megatron.core plus a built model; belongs to the trainer gate, not a unit run"
    ),
    "isoexec/runtimes/vllm/tests/gdn_slot_map_gpu.py": (
        "needs a built vLLM engine and its cumem allocator; belongs to the engine gate"
    ),
}


def _uncollected_battery_files() -> list[str]:
    """Every ``*_gpu.py`` a default ``pytest`` would NOT collect, repo-relative and sorted."""
    found = []
    for path in sorted((_REPO_ROOT / "isoexec").rglob("*_gpu.py")):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        found.append(path.relative_to(_REPO_ROOT).as_posix())
    return found


def test_manifest_covers_every_battery_file():
    """CPU, always runs: no battery file is invisible to both pytest and this manifest."""
    on_disk = _uncollected_battery_files()
    assert len(on_disk) >= 20, f"only {len(on_disk)} battery files found; the scan is probably wrong"
    known = {b.path for b in BATTERIES} | set(NOT_RUN_HERE)
    missing = [p for p in on_disk if p not in known]
    assert not missing, (
        "battery files that a default pytest does not collect and this manifest does not name "
        f"either -- they are evidence nothing runs: {missing}. Add a Battery(...) for each, or an "
        "entry in NOT_RUN_HERE saying why it cannot be driven from here."
    )
    stale = sorted(known - set(on_disk))
    assert not stale, f"manifest names battery files that no longer exist: {stale}"


def test_manifest_entries_are_distinct_and_present():
    paths = [b.path for b in BATTERIES]
    assert len(paths) == len(set(paths)), "a battery is listed twice"
    assert not (set(paths) & set(NOT_RUN_HERE)), "a battery is both run and excluded"
    for battery in BATTERIES:
        assert (_REPO_ROOT / battery.path).is_file(), battery.path
        assert battery.claim.strip(), f"{battery.path} names no claim"
