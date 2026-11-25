"""Every 2-CTA (cta_group::2) tensor-memory allocation site in the tree is a known, audited one.

sm100_common/csrc/tmem_pair.cuh states the four ordering rules (R1 peer started, R2 one writer of
the slot, R3 releasing publish + fences, R4 fenced pair-wide teardown); a pair allocation that breaks
one of them hangs or faults intermittently. A new kernel that allocates tensor memory for a CTA pair must
either use the helper or be added below with the evidence that it follows R1-R4 inline. Static
checks only (no GPU) over IsoExec's own kernel sources (``_vendor`` holds upstream code, out of scope): the
allow-list, no sentinel store into a slot of a pair allocation, and no
relaxed cluster arrive left in a file that moved onto the helper.
"""

import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[2]

#: Pair allocation markers: cute's 2-SM allocator, the raw PTX, and the helper's pair forms.
PAIR = re.compile(r"Allocator2Sm|tcgen05\.alloc\.cta_group::2|tmem::alloc<\s*(2|k\w+)|tmem::setup<\s*(2|k\w+)")

#: path (relative to isoexec/ops) -> how R1-R4 hold.
AUDITED = {
    "mm/mm_sm100/csrc/mm_sm100_leaf.cuh": "base kernel (Qwen path), audited as it stands on release-blackwell; the "
    "R1 ordering change of deepseek-v41-int 6edd9a69 (allocate after a completed cluster barrier) is a separate "
    "base fix, not part of this tree",
    "mm/mm_sm100/general/csrc/gen.cu": "no sentinel; the TMEM base is read after cluster_sync (CL 2) / __syncthreads (CL 1) and tcgen05.fence::after_thread_sync (defect 15 fix 26944d4f)",
    "attention/gqa_sm100/csrc/leaf_d256.cu": "inline: synchronize_pair() (fence, arrive.release, wait.acquire, fence) "
    "before the alloc and after it; same at teardown",
}

#: Base kernels that still store a sentinel into a pair-allocation slot (R2). The fix (deepseek-v41-int 6edd9a69:
#: no sentinel, read the own slot after the fences) touches the Qwen path's GEMM and is tracked as a separate base
#: change; this set must shrink to empty when it lands, and nothing may join it.
KNOWN_R2_PENDING: set = set()
SENTINEL = re.compile(r"tmem\w*\s*\)\s*\)\s*,\s*\"r\"\s*\(\s*0xffffffff", re.I)
SOURCES = (".cu", ".cuh", ".h", ".hpp", ".inl")


def _pair_sites():
    out = {}
    for path in sorted(OPS.rglob("*")):
        if path.suffix not in SOURCES or not path.is_file() or {"sm100_common", "_vendor"} & set(path.parts):
            continue
        text = path.read_text(errors="replace")
        if PAIR.search(text):
            out[str(path.relative_to(OPS))] = text
    return out


def test_every_pair_allocation_is_audited():
    sites = _pair_sites()
    unknown = sorted(set(sites) - set(AUDITED))
    assert not unknown, (
        f"new cta_group::2 tensor-memory allocation(s) {unknown}: use sm100_common/csrc/tmem_pair.cuh "
        "(R1-R4) or add the file to AUDITED with the evidence it follows the rules inline")


def test_no_sentinel_in_pair_allocation_slots():
    bad = [p for p, t in _pair_sites().items() if SENTINEL.search(t)]
    assert set(bad) <= KNOWN_R2_PENDING, f"R2: a sentinel is stored into a tensor-memory slot of a pair allocation in {bad}"
    stale = KNOWN_R2_PENDING - set(bad)
    assert not stale, f"KNOWN_R2_PENDING lists {sorted(stale)}, which no longer store a sentinel: drop them"
    bad = [p for p in bad if p not in KNOWN_R2_PENDING]
    assert not bad, f"R2: a sentinel is stored into a tensor-memory slot of a pair allocation in {bad}"


def test_helper_users_have_no_relaxed_pair_barrier_after_alloc():
    # A file on the helper publishes with cluster_sync_release_acquire / teardown; a relaxed arrive is
    # allowed only as the ENTRY barrier (proves the peer started, publishes nothing).
    for rel, how in AUDITED.items():
        if how != "helper":
            continue
        text = (OPS / rel).read_text()
        for m in re.finditer(r"tmem::alloc<", text):
            tail = text[m.end(): m.end() + 600]
            assert "cluster_arrive_relaxed" not in tail, f"{rel}: relaxed cluster arrive after a pair allocation (R3)"


#: Any tcgen05 allocation (1- or 2-CTA). R3: the base is the slot's value, never column 0.
ANY_ALLOC = re.compile(r"Allocator[12]Sm\b|tcgen05\.alloc\.cta_group|tmem::alloc<|tmem::setup<")
#: A dealloc / teardown of a literal column 0.
ZERO_FREE = re.compile(r"\.free\(\s*0u?\s*,|tmem::teardown<[^>]*>\(\s*0u?\s*,|tmem::dealloc<[^>]*>\(\s*0u?\s*,"
                       r"|tcgen05\.dealloc\.cta_group::\d[^;]*;\"\s*::\s*\"r\"\(\s*0u?\s*\)")
#: Files allowed a literal-0 dealloc: they allocate all 512 columns, which tcgen05.alloc can only satisfy at 0.
ZERO_OK_512 = {"attention/dsa_sm100/csrc/dualrail/leaf.cuh", "attention/dsa_sm100/csrc/dualrail/split.cuh"}


def test_no_hard_coded_tmem_base():
    bad = []
    for path in sorted(OPS.rglob("*")):
        if path.suffix not in SOURCES or not path.is_file() or {"sm100_common", "_vendor"} & set(path.parts):
            continue
        rel = str(path.relative_to(OPS))
        text = path.read_text(errors="replace")
        if ANY_ALLOC.search(text) and ZERO_FREE.search(text) and rel not in ZERO_OK_512:
            bad.append(rel)
    assert not bad, (f"R3: tensor memory freed at a hard-coded column 0 in {bad}; read the base from the slot "
                     "(tmem_pair.cuh read_base) and add it to every TMEM address. A concurrent allocator on the SM "
                     "moves the base (an unspecified launch failure or a hang under concurrent kernels)")


def test_dualrail_zero_free_is_a_512_column_allocation():
    for rel in sorted(ZERO_OK_512):
        text = (OPS / rel).read_text()
        assert "allocate(512," in text and "free(0, 512)" in text, rel
