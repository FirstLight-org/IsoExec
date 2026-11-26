"""Exhaustive CPU model of the sparse_sm100 sparse-attention leaf's mbarrier protocol.

The machine (``Program``/``Model``) is gqa_sm100's ``tests/test_barrier_model_cpu.py``, verbatim:
``try_wait.parity(p)`` passes iff the phase parity != p; an arrival completing the count advances the
phase; a TMA arrival is a pending event that may land at any time in any order; tcgen05.commit
arrivals retire in issue order (``ordered_commits``).  A state where no role can move and some role is
unfinished is a deadlock.

The roles are ``csrc/attn/leaf.cuh``'s, one program each (the four softmax warps modelled as two with
arrival counts 2, the dequant warps as one with counts 1 -- the arrival structure is identical).
Two schedules, ``build(sep=...)``:

  sep=False (in place, 3 stages; also the bf16-kv path's structure, kv_full <-> raw_full):
  gather   (translation prologue) arrive table_ready; per tile t: [t >= 3] wait bar_pv[s] (PV(t-3));
           tma raw_full[s] (the raw rows land in the bf16 stage they become)
  dequant  per tile t: wait raw_full[s]; (named barrier among the dequant warps); arrive kv_full[s][c]
  sep=True (shipped fp8 schedule, SPARSE_RAW_SEP=1: 2 bf16 stages + 2 raw stages):
  gather   per tile t: [t >= 2] wait raw_empty[r] (the dequant of t-2 holds its words); tma raw_full[r]
  dequant  per tile t: wait raw_full[r]; arrive raw_empty[r]; [t >= 2] wait bar_pv[s] (PV(t-2));
           arrive kv_full[s][c]
  mma      wait bar_q; QK(0); per tile g: [g+1 < T] (wait kv_full[(g+1)%2]; commit bar_qk[(g+1)%2]);
           wait bar_p; commit bar_pv[g%2]
  softmax  arrive bar_q; per tile g: [g == 0] wait table_ready; wait bar_qk[g%2];
           [g > 0] wait bar_pv[(g-1)%2]; arrive bar_p; at the end wait bar_pv[last]

Negative controls the model must catch (or it is not sensitive enough to trust): ONE rotating
``bar_pv`` (armed per tile, waited per slot cycle) must deadlock.  ``table_ready`` is shown to be a
visibility (release/acquire) barrier, not a liveness one.
"""

from __future__ import annotations

import pytest

def _wait_ok(state, index, parity):
    _count, phase = state[index]
    return (phase & 1) != parity


def _arrive(state, index, expect):
    count, phase = state[index]
    count += 1
    if count == expect:
        return (0, phase + 1)
    return (count, phase)


class Program:
    """One role's instruction list, with forward branches.

    ``("wait", barrier, parity)``, ``("arrive", barrier)``, ``("tma", barrier)`` (issue a copy whose
    arrival is pending), ``("commit", barrier)`` (issue an MMA fence whose arrival is pending and
    ordered against this role's earlier commits), ``("branch", label)`` (BOTH successors are taken
    by the search: the fall-through and the label) and ``("goto", label)``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.code: list[tuple] = []
        self.labels: dict[str, int] = {}

    def wait(self, barrier: str, parity: int) -> None:
        self.code.append(("wait", barrier, parity & 1))

    def arrive(self, barrier: str) -> None:
        self.code.append(("arrive", barrier))

    def tma(self, barrier: str) -> None:
        self.code.append(("tma", barrier))

    def commit(self, barrier: str) -> None:
        self.code.append(("commit", barrier))

    def branch(self, label: str) -> None:
        self.code.append(("branch", label))

    def goto(self, label: str) -> None:
        self.code.append(("goto", label))

    def label(self, name: str) -> None:
        self.labels[name] = len(self.code)

    def resolve(self) -> None:
        """Turn label names into indices, once the whole program is emitted."""
        self.code = [(op[0], self.labels[op[1]]) if op[0] in ("branch", "goto") else op
                     for op in self.code]


class Model:
    def __init__(self, barriers: dict[str, int], programs: list[Program],
                 ordered_commits: bool = True) -> None:
        self.names = sorted(barriers)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.expect = tuple(barriers[name] for name in self.names)
        self.programs = programs
        self.ordered_commits = ordered_commits
        for program in programs:
            program.resolve()

    def search(self, limit: int = 12_000_000):
        """Depth-first over every interleaving. Returns ``(states, deadlock or None)``."""
        start = (
            tuple(0 for _ in self.programs),                 # program counters
            tuple((0, 0) for _ in self.names),               # (arrivals, phase) per barrier
            (),                                              # pending TMA arrivals (a multiset)
            (),                                              # pending commits, in issue order
        )
        seen = {start}
        stack = [start]
        while stack:
            if len(seen) > limit:
                raise RuntimeError(f"state space above {limit}")
            state = stack.pop()
            pcs = state[0]
            moved = False
            for step in self._successors(state):
                moved = True
                if step not in seen:
                    seen.add(step)
                    stack.append(step)
            if not moved:
                if all(pc == len(p.code) for pc, p in zip(pcs, self.programs)):
                    continue
                return len(seen), self._explain(state)
        return len(seen), None

    def _successors(self, state):
        pcs, bars, tmas, commits = state
        out = []
        for role, program in enumerate(self.programs):
            pc = pcs[role]
            if pc == len(program.code):
                continue
            instruction = program.code[pc]
            kind = instruction[0]

            def jump(target, role=role):
                return pcs[:role] + (target,) + pcs[role + 1:]

            if kind == "wait":
                index = self.index[instruction[1]]
                if _wait_ok(bars, index, instruction[2]):
                    out.append((jump(pc + 1), bars, tmas, commits))
            elif kind == "arrive":
                index = self.index[instruction[1]]
                new = bars[:index] + (_arrive(bars, index, self.expect[index]),) + bars[index + 1:]
                out.append((jump(pc + 1), new, tmas, commits))
            elif kind == "tma":
                out.append((jump(pc + 1), bars, tuple(sorted(tmas + (instruction[1],))), commits))
            elif kind == "commit":
                out.append((jump(pc + 1), bars, tmas, commits + (instruction[1],)))
            elif kind == "goto":
                out.append((jump(instruction[1]), bars, tmas, commits))
            else:  # branch: the warp ballot goes either way
                out.append((jump(pc + 1), bars, tmas, commits))
                out.append((jump(instruction[1]), bars, tmas, commits))
        # an outstanding copy may land at any time, in any order
        for position, name in enumerate(tmas):
            index = self.index[name]
            new = bars[:index] + (_arrive(bars, index, self.expect[index]),) + bars[index + 1:]
            out.append((pcs, new, tmas[:position] + tmas[position + 1:], commits))
        # the MMA pipe is IN ORDER, so only the oldest commit can retire
        retirable = range(1) if self.ordered_commits else range(len(commits))
        for position in retirable:
            if position >= len(commits):
                break
            index = self.index[commits[position]]
            new = bars[:index] + (_arrive(bars, index, self.expect[index]),) + bars[index + 1:]
            out.append((pcs, new, tmas, commits[:position] + commits[position + 1:]))
        return out

    def _explain(self, state) -> str:
        pcs, bars, tmas, commits = state
        lines = []
        for pc, program in zip(pcs, self.programs):
            if pc == len(program.code):
                lines.append(f"{program.name}: finished")
            else:
                lines.append(f"{program.name}: blocked at {program.code[pc]}")
        readable = {name: bars[i] for name, i in self.index.items()}
        return "\n".join(lines) + f"\npending tma={tmas} commits={commits}\nbarriers={readable}"



CHUNKS = 2            # the kernel has 2 chunk groups (stage counts: see build)


def build(tiles: int, warps: int = 2, rotating_pv: bool = False, table_barrier: bool = True, sep: bool = False,
          tile_ready: bool = False):
    """``sep=False``: the in-place form (3 bf16 stages, the raw gather lands in its stage).
    ``sep=True`` (``SPARSE_RAW_SEP=1``): 2 bf16 stages + 2 raw stages; the gather waits ``raw_empty[r]``
    (the dequant warps hold the raw words), the dequant waits ``PV(t - 2)`` before storing."""
    KV = 2 if sep else 3
    RAW = 2
    pv = (lambda s: "bar_pv") if rotating_pv else (lambda s: f"bar_pv{s}")

    def pv_parity(t):                  # the parity a waiter for PV(t)'s completion waits on
        # the negative control is the one-bit-clock bug itself: a barrier armed once per TILE (one
        # rotating bar_pv) waited with the per-SLOT-cycle parity the shipped two-barrier form uses
        return (t // KV) & 1

    g = Program("gather")
    # the translation prologue (warps 5-7, a named barrier among themselves), then table_ready
    if table_barrier and not tile_ready:
        g.arrive("table_ready")
    extra = []
    if tile_ready:
        # round 3 (SPARSE_TILE_READY): two translators publish alternate tiles; a checker (warp 6) waits all
        # of them and publishes check_done; gather and softmax wait tile_ready per tile, the epilogue
        # waits check_done
        for k in range(2):
            tr = Program(f"translator{k}")
            for t in range(k, tiles, 2):
                tr.arrive(f"tile_ready{t}")
            if k == 1:
                for t in range(tiles):
                    tr.wait(f"tile_ready{t}", 0)
                tr.arrive("check_done")
            extra.append(tr)
    for t in range(tiles):
        if tile_ready:
            g.wait(f"tile_ready{t}", 0)
        if sep:
            r = t % RAW
            if t >= RAW:
                g.wait(f"raw_empty{r}", ((t // RAW) - 1) & 1)
            g.tma(f"raw_full{r}")
        else:
            s = t % KV
            if t >= KV:
                g.wait(pv(s), pv_parity(t - KV))          # the stage is free once PV(t - 3) read it
            g.tma(f"raw_full{s}")
    d = Program("dequant")
    for t in range(tiles):
        s = t % KV
        if sep:
            r = t % RAW
            d.wait(f"raw_full{r}", (t // RAW) & 1)
            d.arrive(f"raw_empty{r}")
            if t >= KV:
                d.wait(pv(s), pv_parity(t - KV))
        else:
            d.wait(f"raw_full{s}", (t // KV) & 1)
        for c in range(CHUNKS):
            d.arrive(f"kv_full{s}_{c}")
    mma = Program("mma")
    mma.wait("bar_q", 0)
    for c in range(CHUNKS):
        mma.wait(f"kv_full0_{c}", 0)
    mma.commit("bar_qk0")
    for t in range(tiles):
        if t + 1 < tiles:
            for c in range(CHUNKS):
                mma.wait(f"kv_full{(t + 1) % KV}_{c}", ((t + 1) // KV) & 1)
            mma.commit(f"bar_qk{(t + 1) % 2}")
        mma.wait("bar_p", t & 1)
        mma.commit(pv(t % KV))
    soft = []
    for w in range(warps):
        c = Program(f"softmax{w}")
        c.arrive("bar_q")
        for t in range(tiles):
            if tile_ready:
                c.wait(f"tile_ready{t}", 0)
            elif t == 0 and table_barrier:
                c.wait("table_ready", 0)
            c.wait(f"bar_qk{t % 2}", (t // 2) & 1)
            if t > 0:
                c.wait(pv((t - 1) % KV), pv_parity(t - 1))
            c.arrive("bar_p")
        c.wait(pv((tiles - 1) % KV), pv_parity(tiles - 1))
        if tile_ready:
            c.wait("check_done", 0)
        soft.append(c)
    barriers = {"bar_q": warps, "bar_p": warps, "bar_qk0": 1, "bar_qk1": 1, "table_ready": 1}
    for i in range(RAW if sep else KV):
        barriers[f"raw_full{i}"] = 1
        if sep:
            barriers[f"raw_empty{i}"] = 1
    for i in range(KV):
        for c in range(CHUNKS):
            barriers[f"kv_full{i}_{c}"] = 1
    if rotating_pv:
        barriers["bar_pv"] = 1
    else:
        for i in range(KV):
            barriers[f"bar_pv{i}"] = 1
    if tile_ready:
        barriers["check_done"] = 1
        for t in range(tiles):
            barriers[f"tile_ready{t}"] = 1
    return Model(barriers, [g, d, mma, *soft, *extra])


@pytest.mark.parametrize("sep", [False, True])
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 5, 6, 7])
def test_cpu_leaf_protocol_deadlock_free(tiles, sep):
    states, dead = build(tiles, sep=sep).search()
    assert dead is None, dead
    assert states > 0


@pytest.mark.parametrize("sep", [True, False])
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 5])
def test_cpu_leaf_protocol_tile_ready(tiles, sep):
    """Round 3's shipped schedules: per-tile table publication on the separate-raw fp8 schedule (sep=True)
    and on the 3-stage bf16 structure (sep=False)."""
    states, dead = build(tiles, sep=sep, tile_ready=True).search()
    assert dead is None, dead
    assert states > 0


@pytest.mark.parametrize("sep", [False, True])
@pytest.mark.parametrize("tiles", [1, 3, 5])
def test_cpu_leaf_protocol_deadlock_free_any_commit_order(tiles, sep):
    """Commit ordering is not load-bearing for liveness."""
    model = build(tiles, sep=sep)
    model.ordered_commits = False
    _, dead = model.search()
    assert dead is None, dead


def test_cpu_rotating_pv_is_caught():
    """One bar_pv armed per tile but waited per slot cycle: the one-bit clock.  Must deadlock."""
    found = False
    for tiles in range(2, 7):
        _, dead = build(tiles, rotating_pv=True).search()
        if dead is None:
            _, dead = build(tiles, rotating_pv=True, sep=True).search()
        if dead is not None:
            found = True
            break
    assert found, "the model did not catch the rotating retire barrier"


def test_cpu_table_ready_is_a_visibility_barrier():
    """Without table_ready no interleaving lets a softmax warp pass tile 0 before the gather warp has
    issued tile 0 (the gather -> dequant -> MMA -> bar_qk chain already orders it), so table_ready is
    not load-bearing for ORDER or liveness.  It is kept because the table is written with generic
    stores by warps 5-7 and read by the softmax warps: the mbarrier's release/acquire is what makes
    those stores visible, and this model has no memory semantics to see that."""
    model = build(3, table_barrier=False)
    assert model.search()[1] is None
    start = ((0,) * len(model.programs), tuple((0, 0) for _ in model.names), (), ())
    seen, stack = {start}, [start]
    while stack:
        st = stack.pop()
        assert not (st[0][0] == 0 and st[0][3] >= 2)
        for nxt in model._successors(st):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)


@pytest.mark.parametrize("tiles", [1, 2, 3])
def test_cpu_leaf_protocol_split_softmax_arrivals(tiles):
    """SM_SPLIT: four softmax programs arriving bar_p / bar_q (count 4 here for the model's halved roles;
    the kernel's count 8).  The intra-pair named barriers (bar.sync 3 + qd, 64) are not mbarriers and are
    not modelled: every half executes the same two per tile unconditionally (README "Round 3")."""
    states, dead = build(tiles, warps=4, sep=False, tile_ready=True).search()
    assert dead is None, dead
