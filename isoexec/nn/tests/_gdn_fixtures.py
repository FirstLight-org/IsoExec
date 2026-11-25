"""Shared tiny GDN and dense-MLP geometry for independent CPU comparisons. Weight mirroring and bit
comparison reuse the attention helpers.
"""

from __future__ import annotations

from isoexec.nn.gdn import GDN
from isoexec.nn.mlp import MLP
from isoexec.schema import GDN as GdnSpec
from isoexec.schema import Axis, Mlp, Placement

from isoexec.nn.tests._fixtures import GROUP, bits, mirror_weights  # noqa: F401  (re-exported for the proof files)

#: The tiny Qwen3.5 linear shape (``models/qwen3_5/tests/tiny.py::TINY_TEXT_COMMON``): 4 key
#: heads over 8 value heads at head_dim 32, conv kernel 4. ``value_heads != key_heads`` is the
#: shipped geometry, not an edge case, and it is what makes the segmented conv axis load-bearing.
HIDDEN = 256
KEY_HEADS = 4
VALUE_HEADS = 8
KEY_HEAD_DIM = 32
VALUE_HEAD_DIM = 32
CONV_KERNEL = 4
INTERMEDIATE = 512
EPS = 1e-6
NS = "lm.layers.0"
SITE = f"{NS}.mixer"
MLP_SITE = f"{NS}.mlp"


def spec(
    *,
    key_heads: int = KEY_HEADS,
    value_heads: int = VALUE_HEADS,
    key_head_dim: int = KEY_HEAD_DIM,
    value_head_dim: int = VALUE_HEAD_DIM,
    conv_kernel: int = CONV_KERNEL,
) -> GdnSpec:
    return GdnSpec(
        num_key_heads=key_heads,
        num_value_heads=value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel=conv_kernel,
    )


def mlp_spec(*, intermediate: int = INTERMEDIATE) -> Mlp:
    return Mlp(intermediate_size=intermediate)


def component(row: GdnSpec, *, device: str = "meta") -> GDN:
    """Build GDN with the same fields as the schema row. Chunk length remains owned by the
    resolved core.
    """
    return GDN(
        hidden=HIDDEN,
        key_heads=row.num_key_heads,
        value_heads=row.num_value_heads,
        key_head_dim=row.key_head_dim,
        value_head_dim=row.value_head_dim,
        conv_kernel=row.conv_kernel,
        out_norm=row.out_norm,
        eps=EPS,
        device=device,
    )


def mlp_component(row: Mlp, *, device: str = "meta") -> MLP:
    return MLP(hidden=HIDDEN, intermediate=row.intermediate_size, activation=row.activation, device=device)


def placement(tp: int = 1, rank: int = 0, *, side: str = "trainer", cp: int = 1) -> Placement:
    """A ``Placement`` at the TP (and CP) degree under test; every other axis is 1."""
    one = Axis(1, 0, None)
    tp_axis = Axis(tp, rank, GROUP if tp > 1 else None)
    return Placement(
        tp=tp_axis,
        ep=one,
        dp=one,
        pp=one,
        etp=tp_axis if tp > 1 else one,
        cp=Axis(cp, 0, GROUP if cp > 1 else None),
        side=side,
        leaves=8,
    )
