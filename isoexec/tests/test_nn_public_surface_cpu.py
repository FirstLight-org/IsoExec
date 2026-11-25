"""An external component binds arithmetic and placement seams through the public API."""

from __future__ import annotations
from isoexec.ops.table import OpTable as ImplementationTable

import ast
import pathlib

import pytest
import torch
from torch import nn

from isoexec.nn import (
    NO_MOVEMENT,
    Component,
    PlacementError,
    BoundOp,
    SeamError,
    Seams,
    place,
    resolved_ops,
)
from isoexec.schema import Axis, Placement, col, row

HIDDEN = 64
INNER = 128
SITE = "lm.layers.0.mixer"


def placement(tp: int = 1, *, side: str = "trainer", sp: bool = False, leaves: int = 8) -> Placement:
    """A ``Placement`` at the degree under test, spelled with public schema names only."""
    one = Axis(1, 0, None)
    group = object() if tp > 1 else None  # these tests never run a collective; a placement CARRIES one
    tp_axis = Axis(tp, 0, group)
    return Placement(
        # etp follows tp at ep=1: every rank tensor-shards every expert over the TP group.
        tp=tp_axis, ep=one, dp=one, pp=one, etp=tp_axis if tp > 1 else one, cp=one, side=side, leaves=leaves, sp=sp
    )


class ResidualProjector(Component):
    """External component using only public nn APIs for projections and placement seams."""

    SLOT_OPS = {"mm": "mm", "tp_copy": "collectives.tp_copy", "row_parallel": "collectives.row_parallel"}
    SLOTS = tuple(SLOT_OPS)

    def __init__(self, *, hidden: int = HIDDEN, inner: int = INNER, device: torch.device | str = "meta") -> None:
        super().__init__()
        self.hidden, self.inner = hidden, inner
        self.up = nn.Parameter(torch.empty((inner, hidden), dtype=torch.bfloat16, device=device))
        self.down = nn.Parameter(torch.empty((hidden, inner), dtype=torch.bfloat16, device=device))

    def tensor_axes(self):
        return {"up": col(units=self.inner), "down": row()}

    def facts(self):
        return {"kind": "residual_projector", "hidden": self.hidden, "inner": self.inner, "held_dtype": "bf16"}

    def _place(self, placement: Placement) -> None:
        seams = self.seams
        self.slots.tp_copy = seams.tp_copy()
        self.slots.row_parallel = seams.row_parallel()
        self.slots.mm = self.bind_op("mm", site=f"{self.site}.projections")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mm = self.slots.mm
        return self.slots.row_parallel(mm(self.slots.tp_copy(x), self.up), self.down)


def built(tp: int = 1, **kw) -> ResidualProjector:
    module = ResidualProjector()
    place(module, placement(tp=tp, **kw), prefix=SITE, op_table=ImplementationTable(mode='reference'))
    return module


# the surface is reachable


def test_an_out_of_tier_component_fills_every_slot_with_a_real_resolution():
    """Public APIs must fill every slot with production provenance."""
    slots = resolved_ops(built())[SITE]
    assert set(slots) == set(ResidualProjector.SLOTS)
    assert all(isinstance(r, BoundOp) for r in slots.values())
    assert "test" not in {r.provider for r in slots.values()}
    assert slots["row_parallel"].op == "collectives.row_parallel"
    assert slots["tp_copy"].op == "collectives.tp_copy"


def test_a_seam_that_moves_nothing_is_recorded_rather_than_absent():
    """TP=1 seams retain an explicit NO_MOVEMENT binding."""
    slots = resolved_ops(built(tp=1))[SITE]
    assert slots["tp_copy"].impl == NO_MOVEMENT
    assert slots["row_parallel"].impl != NO_MOVEMENT


def test_the_whole_seam_surface_is_reachable_without_a_component():
    """Every seam, built directly from a ``Placement`` -- the form a host walk or a test takes.

    ``Component.seams`` is the convenience; :class:`Seams` is the surface. A caller that has a
    placement and no component (a degree sweep proving what tp changes, a fixture) constructs one.
    """
    seams = Seams(placement(tp=1), SITE, ImplementationTable(mode="reference"))
    for built_seam in (seams.tp_copy(), seams.tp_grad_sum(), seams.kv_gather(kv_heads=2), seams.sp_gather(),
                       seams.sp_scatter()):
        assert isinstance(built_seam, BoundOp) and built_seam.impl == NO_MOVEMENT
    assert isinstance(seams.row_parallel(), BoundOp)
    assert isinstance(seams.row_parallel(inside_sp_gather=True), BoundOp)
    # Context-parallel construction returns gather and core together; cp=1 preserves the
    # original core.
    gather, core = seams.cp_attention(None)
    assert isinstance(gather, BoundOp) and gather.impl == NO_MOVEMENT and core is None


def test_at_narrows_the_site_so_two_contractions_get_two_names():
    """``seams.at("fc1")`` is how a component whose PARTS take different seams says which part."""
    seams = Seams(placement(tp=1), SITE, ImplementationTable(mode="reference"))
    assert seams.at("fc1").site == f"{SITE}.fc1"
    assert seams.at("fc1").at("shared").site == f"{SITE}.fc1.shared"
    assert seams.at("fc1").placement is seams.placement
    with pytest.raises(SeamError, match="non-empty part"):
        seams.at("")


def test_a_refusal_from_a_seam_names_the_site_the_component_was_placed_at():
    """The site is why the seam takes one: a refusal has to say WHERE, and only the walk knows."""
    with pytest.raises(SeamError, match=rf"{SITE}\.row_parallel: tp=4 must divide leaves=2"):
        Seams(placement(tp=4, leaves=2), SITE, ImplementationTable(mode="reference")).row_parallel()


def test_seams_are_unavailable_before_placement_and_say_why():
    """There is nothing to build one from: a seam is a function of a ``Placement``."""
    with pytest.raises(PlacementError, match="apply_placement has not run"):
        _ = ResidualProjector().seams


# the surface is the WHOLE surface


def test_this_module_imports_nothing_private():
    """The external component fixture must import only public IsoExec names."""
    tree = ast.parse(pathlib.Path(__file__).read_text())
    private: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isoexec"):
            if any(part.startswith("_") for part in (node.module or "").split(".")):
                private.append(node.module or "")
            private += [a.name for a in node.names if a.name.startswith("_")]
        elif isinstance(node, ast.Import):
            private += [a.name for a in node.names if a.name.startswith("isoexec") and "._" in a.name]
    assert private == [], f"this file reaches into private modules for: {private}"
