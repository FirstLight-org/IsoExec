"""Admission of the V4.1 CED definition: every layout it admits or refuses, in one function.

``admit(cfg, placement, site, ...)`` is called by every placed component of the model (encoder blocks,
``decoder.entry``, decoder blocks) and by the runtime bindings before they build anything. Each refusal is a
``Refusal`` whose ``code`` names it; a lane that lifts a restriction edits this function only.

Admitted: TP in {1, 2, 4, 8} (aligned subtrees of the fixed 8-leaf trees); EP dividing the routed experts,
over the TP ranks (EP = TP) or token-disjoint at TP = 1 on the engine and on the trainer (the recipe's trainer
layout, TP1 with EP over all GPUs; GPU pending: ``moe/deepseek_v41_sm100`` ``ep_moe_train`` sums the router's and
the shared expert's gradients over the EP group, so Megatron reduces them as expert tensors); PP at the admissible stage starts of the
geometry (``CedGeometry.admissible_pipeline_cuts``: the encoder's SWA layers and group starts, and the CED
boundary, where only the residual carry crosses); both CED prefill programs. Refused, by name: below.
"""
from __future__ import annotations

from collections.abc import Sequence

from isoexec.nn.component import PlacementError

from .config import CED_PREFILL_MODES, CedGeometry

TP_TREE = "TP_TREE"
TP_HEAD_SHARD = "TP_HEAD_SHARD"
EP_EXPERT_SPLIT = "EP_EXPERT_SPLIT"
EP_ETP = "EP_ETP"
EP_GROUP = "EP_GROUP"
SEQUENCE_PARALLEL = "SEQUENCE_PARALLEL"
CONTEXT_PARALLEL = "CONTEXT_PARALLEL"
PP_STAGES_UNDECLARED = "PP_STAGES_UNDECLARED"
PP_CUT_INSIDE_DECODER = "PP_CUT_INSIDE_DECODER"
PP_CUT_SPLITS_GROUP = "PP_CUT_SPLITS_GROUP"
PROGRAM_UNKNOWN = "PROGRAM_UNKNOWN"


class Refusal(PlacementError):
    """A layout the V4.1 definition refuses; ``code`` is one of this module's names."""

    def __init__(self, code: str, site: str, message: str) -> None:
        super().__init__(f"{site}: {code}: {message}")
        self.code = code


def admit_pipeline(cfg: CedGeometry, stage_starts: Sequence[int], site: str = "model") -> None:
    """Stage ``k`` starts at layer ``stage_starts[k]`` (the first is 0). Every later start must be an admissible
    cut: no cut inside the decoder (every decoder layer reads the decoder global KV owned by ``decoder.entry``),
    none between an encoder FULL layer and the REUSE layers that read its compressed KV and top-k."""
    starts = list(stage_starts)
    if not starts or starts[0] != 0 or starts != sorted(set(starts)) or starts[-1] >= cfg.num_hidden_layers:
        raise Refusal(PP_STAGES_UNDECLARED, site, f"stage starts {starts} must ascend from 0 inside the backbone")
    ok = set(cfg.admissible_pipeline_cuts())
    for s in starts[1:]:
        if s > cfg.boundary:
            raise Refusal(PP_CUT_INSIDE_DECODER, site, f"a stage starting at decoder layer {s} would separate it "
                                                       f"from the decoder global KV that decoder.entry (layer "
                                                       f"{cfg.boundary}) owns; admissible starts {sorted(ok)}")
        if s not in ok:
            raise Refusal(PP_CUT_SPLITS_GROUP, site, f"a stage starting at layer {s} separates a REUSE layer from "
                                                     f"its FULL source's compressed KV and top-k (native vLLM "
                                                     f"refuses the same split); admissible starts {sorted(ok)}, "
                                                     f"VLLM_PP_LAYER_PARTITION over {len(starts)} stages one of "
                                                     f"{admissible_partitions(cfg, len(starts))} (layers per stage; "
                                                     f"the CED cut at PP2: {cfg.boundary},"
                                                     f"{cfg.num_hidden_layers - cfg.boundary})")


def even_partition(count: int, pp: int) -> list[int]:
    """vLLM's default split of ``count`` layers over ``pp`` stages (``vllm.distributed.utils.get_pp_indices``
    without ``VLLM_PP_LAYER_PARTITION``): equal parts, the remainder added one each to the stages before the
    last, from the second-to-last backwards (``partitions[-i] += 1`` for ``i`` in ``2 .. remainder + 1``)."""
    base, extra = divmod(count, pp)
    counts = [base] * pp
    for i in range(2, extra + 2):
        counts[-i] += 1
    return counts


def layer_partition(cfg: CedGeometry, pp: int, partition: str | None = None) -> tuple[tuple[int, int], ...]:
    """Native vLLM's stage layout of the backbone: half-open LAYER ranges per stage, as native DeepSeek V4.1 lays
    out its ``num_hidden_layers`` layers (``make_layers`` -> ``get_pp_indices``): ``VLLM_PP_LAYER_PARTITION``
    counts layers per stage, as for native; without it vLLM's even split (``even_partition``). At production depth
    (40 layers) the PP2 default is 20 / 20, the CED cut."""
    n = cfg.num_hidden_layers
    if partition:
        counts = [int(c) for c in partition.split(",")]
        if len(counts) != pp or sum(counts) != n or min(counts) < 1:
            raise Refusal(PP_STAGES_UNDECLARED, "model", f"VLLM_PP_LAYER_PARTITION {partition!r} must split the "
                                                         f"{n} backbone layers over {pp} stages")
    else:
        counts = even_partition(n, pp)
        if min(counts) < 1:
            raise Refusal(PP_STAGES_UNDECLARED, "model", f"{n} backbone layers do not fill {pp} stages")
    out, lo = [], 0
    for c in counts:
        out.append((lo, lo + c))
        lo += c
    return tuple(out)


def vllm_layer_partition(cfg: CedGeometry, pp: int) -> tuple[tuple[int, int], ...]:
    """``layer_partition`` as vLLM computes it in this process (``get_pp_indices`` over the backbone layers, the
    environment's ``VLLM_PP_LAYER_PARTITION`` included): the same ranges vLLM's own config uses for the stage
    (``ModelConfig.get_layers_start_end_indices``)."""
    from vllm.distributed.utils import get_pp_indices
    return tuple(get_pp_indices(cfg.num_hidden_layers, rank, pp) for rank in range(pp))


def stage_starts_from_partition(cfg: CedGeometry, pp: int, partition: str | None = None) -> tuple[int, ...]:
    """The stage starts (backbone layer ids) of ``layer_partition``."""
    return tuple(lo for lo, _ in layer_partition(cfg, pp, partition))


def unit_ranges(cfg: CedGeometry, layer_ranges) -> tuple[tuple[int, int], ...]:
    """Admit the stage layout of ``layer_ranges`` (``admit_pipeline``: a cut inside the decoder or inside a CSA2
    group is refused by name) and map it onto the pipeline units the stage split takes (the encoder blocks, unit
    ``u < b`` = layer ``u``, then the decoder as unit ``b``)."""
    starts = [lo for lo, _ in layer_ranges]
    admit_pipeline(cfg, starts)
    b = cfg.boundary
    return tuple((lo, hi if hi <= b else b + 1) for lo, hi in layer_ranges)


def admissible_partitions(cfg: CedGeometry, pp: int) -> list[str]:
    """Every ``VLLM_PP_LAYER_PARTITION`` value (layers per stage) over ``pp`` stages whose cuts are admissible."""
    n, cuts = cfg.num_hidden_layers, cfg.admissible_pipeline_cuts()
    out = []

    def walk(prefix, start):
        if len(prefix) == pp - 1:
            out.append(",".join(str(c) for c in (*prefix, n - start)))
            return
        for s in cuts:
            if s > start and n - s >= pp - 1 - len(prefix):
                walk((*prefix, s - start), s)
    walk((), 0)
    return out


def admit(cfg: CedGeometry, placement, site: str, *, program: str | None = None,
          stage_starts: Sequence[int] | None = None) -> None:
    """Refuse, by name, a layout this definition does not execute."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100.spec import TP_LEAVES

    from .providers import vocab_shard_is_bit_neutral

    if program is not None and program not in CED_PREFILL_MODES:
        raise Refusal(PROGRAM_UNKNOWN, site, f"CED prefill program {program!r} is not one of {CED_PREFILL_MODES}")
    tp, ep = placement.tp.size, placement.ep.size
    if TP_LEAVES % tp or cfg.num_attention_heads % tp or cfg.o_groups % tp:
        raise Refusal(TP_TREE, site, f"TP = {tp} must own aligned subtrees of the fixed {TP_LEAVES}-leaf wo_b and "
                                     f"index trees and divide heads and output groups")
    if not vocab_shard_is_bit_neutral(cfg, tp):
        raise Refusal(TP_HEAD_SHARD, site, f"at TP = {tp} the FP32 head's vocab shard is not a weight mm_sm100 "
                                           f"serves (its logits would not be the replicated head's)")
    if placement.pp.size > 1:
        if stage_starts is None:
            raise Refusal(PP_STAGES_UNDECLARED, site, "PP > 1: the binding must declare its stage starts "
                                                      "(admit_pipeline)")
        admit_pipeline(cfg, stage_starts, site)
    if ep > 1 and cfg.n_routed_experts % ep:
        raise Refusal(EP_EXPERT_SPLIT, site, f"{cfg.n_routed_experts} routed experts do not split over EP = {ep}")
    if ep > 1 and placement.etp.size != 1:
        raise Refusal(EP_ETP, site, f"EP > 1 holds whole experts (ETP = 1), got ETP = {placement.etp.size}")
    if ep > 1 and tp not in (1, ep):
        raise Refusal(EP_GROUP, site, f"EP over the TP ranks (EP = TP) or over token-disjoint ranks (TP = 1) only, "
                                      f"got TP = {tp}, EP = {ep}")
    if getattr(placement, "sp", False):
        raise Refusal(SEQUENCE_PARALLEL, site, "sequence parallelism is not part of the V4.1 recipe (TP1 trainer)")
    if placement.cp.size > 1:
        raise Refusal(CONTEXT_PARALLEL, site, "the CP schedule of the encoder windows, compressed KV and indexer is "
                                              "not designed yet (axes declared, briefs/CP.md)")


__all__ = ["Refusal", "admissible_partitions", "admit", "admit_pipeline", "even_partition", "layer_partition",
           "stage_starts_from_partition", "unit_ranges", "vllm_layer_partition"]
