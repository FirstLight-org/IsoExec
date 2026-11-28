"""CPU checks of the fused phased transport (mode 9, csrc tree_fused_kernel): launch plan, index maps, progress, numerics.

Everything here is W-generic and runs for TP 2 / 4 / 8 (and 16) without GPUs:
* fused_plan covers every element exactly once with legal launches (the kernel's TORCH_CHECKs re-implemented);
* a model of the kernel's address maps (phase A pushes, B folds, C gathers) shows every pushed word has one writer,
  is read exactly once by its owner from the same item, stays inside the sender slot, and every output element is
  written exactly once on every rank;
* a scheduling model with adversarial per-rank residency shows the dense slot ticket always completes (and that the
  model is not vacuous: matching-blockIdx assignment deadlocks in it);
* the fixed tree over W subtree roots equals the 8-leaf tree for every W | 8 (TP invariance by construction);
* source checks: same tree / rounding / sentinel helpers as mode 8, bounded waits, no PDL in the fused kernel.
"""
from __future__ import annotations

import itertools
import random
import re
from pathlib import Path

import numpy as np
import pytest

from isoexec.ops.collectives.detreduce_sm100.phased import TILE, fused_plan

CU = (Path(__file__).resolve().parents[1] / "csrc" / "detreduce.cu").read_text()
LCAP = LYCAP = 4 << 20


def tree_slot(cap, world):
    return (cap // world) // TILE * TILE


def two_shot_chunk(lcap, lycap, world, grid, unroll):
    unit = grid * unroll * TILE
    return min(lycap, tree_slot(lcap, world) // unit * world * unit)


def legal(n, two, grid, unroll, world, lcap=LCAP, lycap=LYCAP):
    item = unroll * TILE
    items = -(-n // item)
    assert n % 4 == 0 and n > 0 and unroll in (1, 4)
    assert 1 <= grid <= items and grid <= 1024
    lslot = tree_slot(lcap, world)
    if two:
        rounds = -(-items // grid)
        assert grid * (-(-rounds // world)) * item <= lslot and n <= lycap
    else:
        assert n <= lslot


SIZES = [4, 1020, 1024, 1028, 5120, 8 * 5120, 101 * 5120, 102 * 5120, 103 * 5120, 256 * 5120, 409 * 5120,
         2048 * 5120, 4096 * 5120, 37 * (1 << 20) + 12, 4096, 128 * 7168]


@pytest.mark.parametrize("world", [2, 4, 8, 16])
@pytest.mark.parametrize("max_grid", [1, 7, 148, 1024])
def test_plan_covers_and_is_legal_cpu(world, max_grid):
    for n in SIZES:
        plan = fused_plan(n, world, tree_slot(LCAP, world), LYCAP, LCAP, max_grid, 512 << 10, two_shot_chunk)
        assert plan == fused_plan(n, world, tree_slot(LCAP, world), LYCAP, LCAP, max_grid, 512 << 10, two_shot_chunk)
        pos = 0
        for start, stop, two, grid, unroll in plan:
            assert start == pos and stop > start and start % 4 == 0
            legal(stop - start, two, grid, unroll, world)
            pos = stop
        assert pos == n
        assert len({two for *_, two, _, _ in plan}) == 1      # one form per call (decided from n and W only)


def test_plan_small_m_is_one_small_launch_cpu():
    for world in (2, 4, 8):
        for M in (1, 8):
            plan = fused_plan(M * 5120, world, tree_slot(LCAP, world), LYCAP, LCAP, 148, 512 << 10, two_shot_chunk)
            assert len(plan) == 1 and not plan[0][2]
            assert plan[0][3] == -(-(M * 5120) // TILE) and plan[0][4] == 1   # grid = tiles, one tile per CTA


def simulate_maps(world, n, grid, unroll, two, lslot):
    """Replays the kernel's index arithmetic (tree_fused_kernel) for every rank / slot / item at tile granularity (every
    thread of a tile shifts element and box offset by the same t * 4); asserts the map properties."""
    item = unroll * TILE
    items = -(-n // item)
    writes, reads = {}, {}   # (dst_rank, sender, box offset) -> (element c, length)
    outs = [np.zeros(n, dtype=np.int64) for _ in range(world)]
    for me in range(world):
        for j in range(grid):
            for k, i in enumerate(range(j, items, grid)):
                owner = (k + j) % world
                comp = (j + (k // world) * grid) * item if two else i * item
                for u in range(unroll):
                    c = i * item + u * TILE
                    if c >= n:
                        break
                    ln = min(TILE, n - c)
                    off = comp + u * TILE
                    assert off + ln <= lslot, "word outside the sender slot"
                    for q in range(world):          # phase A
                        if q != me and (not two or q == owner):
                            assert (q, me, off) not in writes, "two writers"
                            writes[(q, me, off)] = (c, ln)
                    if not two or owner == me:      # phase B
                        for q in range(world):
                            if q != me:
                                assert (me, q, off) not in reads
                                reads[(me, q, off)] = (c, ln)
                    outs[me][c:c + ln] += 1         # B (own fold) or C (gather of the owner's BF16 words at c / 2)
    assert writes == reads, "every pushed word is read once, by its owner, for the same element"
    for o in outs:
        assert (o == 1).all(), "every output element written exactly once per rank"


@pytest.mark.parametrize("world", [2, 4, 8])
def test_index_maps_cpu(world):
    lslot = 1 << 22
    for n, grid, unroll in [(4, 1, 1), (1028, 1, 1), (5120, 5, 1), (5120, 2, 1), (40960, 40, 1), (40960, 7, 4),
                            (1 << 18, 11, 4), ((1 << 18) + 12, 13, 4)]:
        for two in (False, True):
            simulate_maps(world, n, grid, unroll, two, lslot)


def run_schedule(world, grid, rounds, residency, two, ticket, seed):
    """One call on every rank. CTAs: resident on their rank only while a residency slot is free (taken in launch order,
    as the block scheduler does); each step advances one random runnable CTA by one phase. True if all finish."""
    rnd = random.Random(seed)
    phases = ["A", "B", "C"] if two else ["A", "B"]
    state = {}                     # (rank, cta) -> phase index; slot -> phase done per (rank, slot)
    slot_of = {}
    done_phase = {}                # (rank, slot) -> number of completed phases
    pending = {r: list(range(grid)) for r in range(world)}
    if not ticket:
        for r in range(world):
            rnd.shuffle(pending[r])          # blockIdx start order differs per rank
    resident = {r: [] for r in range(world)}
    next_ticket = {r: 0 for r in range(world)}
    finished = 0
    for _ in range(100_000):
        for r in range(world):
            while pending[r] and len(resident[r]) < residency[r]:
                b = pending[r].pop(0)
                resident[r].append(b)
                if ticket:
                    slot_of[(r, b)] = next_ticket[r]
                    next_ticket[r] += 1
                else:
                    slot_of[(r, b)] = b
                state[(r, b)] = 0
                done_phase[(r, slot_of[(r, b)])] = 0
        runnable = []
        for r in range(world):
            for b in resident[r]:
                j = slot_of[(r, b)]
                p = state[(r, b)]
                need = p   # phase p waits for every peer's slot j to have completed phase p - 1 (A waits nothing)
                if p == 0 or all(q == r or done_phase.get((q, j), -1) >= need for q in range(world)):
                    runnable.append((r, b))
        if not runnable:
            return finished == world * grid
        r, b = rnd.choice(runnable)
        j = slot_of[(r, b)]
        state[(r, b)] += 1
        done_phase[(r, j)] = state[(r, b)]
        if state[(r, b)] == len(phases):
            resident[r].remove(b)
            finished += 1
    raise AssertionError("no fixpoint")


@pytest.mark.parametrize("world", [2, 4, 8])
def test_slot_ticket_progress_any_residency_cpu(world):
    for seed in range(60):
        rnd = random.Random(seed)
        grid = rnd.choice([1, 3, 8, 33])
        residency = [rnd.randint(1, grid) for _ in range(world)]
        for two in (False, True):
            assert run_schedule(world, grid, 1, residency, two, ticket=True, seed=seed)


def test_model_detects_blockidx_deadlock_cpu():
    # the same model with blockIdx-matched slots (the single-kernel modes 5 / 7) can deadlock under partial residency
    hung = 0
    for seed in range(200):
        if not run_schedule(2, 8, 1, [2, 2], False, ticket=False, seed=seed):
            hung += 1
    assert hung > 0


def f32(x):
    return np.float32(x)


def tree(vals):
    v = list(vals)
    while len(v) > 1:
        v = [np.float32(v[2 * k] + v[2 * k + 1]) for k in range(len(v) // 2)]
    return v[0]


def test_tree_over_ranks_is_the_leaf_tree_for_every_tp_cpu():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        leaves = (rng.standard_normal(8) * np.exp2(rng.integers(-30, 30, 8))).astype(np.float32)
        full = tree(leaves)
        for world in (1, 2, 4, 8):
            per = 8 // world
            roots = [tree(leaves[r * per:(r + 1) * per]) for r in range(world)]
            assert tree(roots).tobytes() == full.tobytes()


def _fused_source():
    m = re.search(r"void tree_fused_kernel\(.*?\n}\n", CU, re.S)
    assert m
    return m.group(0)


def test_fused_kernel_uses_the_mode8_arithmetic_cpu():
    src = _fused_source()
    assert "tree4<W>(v[u])" in src and "lamport_pair(r4.x, r4.y)" in src and "lamport_pair(r4.z, r4.w)" in src
    assert "st_lamport4(" in src and "lamport_settle4(" in src
    assert "__fadd" not in src and "atomicAdd(word, 1ull)" in src
    assert "float" not in re.sub(r"float4|float\*|const float\*", "", "".join(re.findall(r"atomic\w+\([^;]*;", src)))


def test_fused_kernel_waits_are_bounded_and_no_pdl_cpu():
    src = _fused_source()
    assert "griddepcontrol" not in src
    assert "spin_until(" in src and "lamport_settle4(" in src and "__trap()" in src
    loops = re.findall(r"while \(", src)
    assert len(loops) == 1          # the only raw wait loop (result words) carries its own 20 s timeout
    assert "20000000000ull" in src
