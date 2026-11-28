"""Token embedding, four-stream expansion and the Engram n-gram hashes of one forward."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from isoexec.nn.component import Component
from isoexec.schema import Placement, ShardAxis, replicated

from .config import Geometry
from .providers import engram_hash_constants


class DeepSeekV41Embed(Component):
    """``input_ids [B, T]`` -> ``x [B, T, 4, H]`` bf16 streams, the one-hot shifted ``pre [B, T, 4]`` fp32 and,
    for a tower with Engram rows, the bucket ids ``hashes [B, T, L, (N-1) heads]`` int64 of the released hash
    layout.

    Without autograd one fused launch gathers, expands and writes ``pre`` (``fused_sm100.embed_expand``,
    bitwise the composed form); with a trainable table it is the embedding, the expand and the one-hot.
    The hash needs the tokenizer's compressed token map (``bind_token_map``); it is not a checkpoint tensor.
    """

    def __init__(self, cfg: Geometry, *, device="meta") -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty((cfg.vocab_size, cfg.hidden_size), dtype=torch.bfloat16,
                                               device=device), requires_grad=False)
        self.hashing = bool(cfg.engram_layer_ids)
        self._hash: dict = {}

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {"weight": replicated()}

    def facts(self) -> Mapping[str, Any]:
        cfg = self.cfg
        return {"kind": "deepseek_v41_embed", "vocab": cfg.vocab_size, "hidden": cfg.hidden_size,
                "streams": cfg.hc_mult, "engram_hash_layers": cfg.released_engram_layers if self.hashing else (),
                "ngram": cfg.engram_max_ngram_size, "hash_heads": cfg.engram_n_heads}

    def _place(self, placement: Placement) -> None:
        # The actor recipe trains the embedding on the trainer; the engine's copy is read-only.
        self.weight.requires_grad_(placement.side == "trainer")

    def bind_token_map(self, token_map) -> None:
        """The tokenizer's compressed token map (one compressed id per tokenizer id, the checkpoint's pinned
        normalization), which the n-gram hash reads."""
        token_map = torch.as_tensor(token_map, dtype=torch.int64)
        cfg = self.cfg
        if token_map.ndim != 1 or token_map.numel() == 0 or int(token_map.min()) < 0:
            raise ValueError("token_map must contain nonnegative compressed IDs")
        if sorted(token_map.unique().tolist()) != list(range(cfg.engram_compressed_vocab_size)):
            raise ValueError("compressed token map vocabulary does not match the checkpoint")
        if not 0 <= cfg.engram_pad_token_id < token_map.numel():
            raise ValueError("Engram pad token must be in the tokenizer vocabulary")
        primes, offsets, multipliers = engram_hash_constants(cfg)
        self._hash = {"cpu": (token_map, primes, offsets, multipliers, int(token_map[cfg.engram_pad_token_id]))}

    def _hash_tensors(self, device):
        if "cpu" not in self._hash:
            raise RuntimeError("Engram hashing needs bind_token_map(token_map) before the first forward")
        key = str(device)
        if key not in self._hash:
            *tensors, pad = self._hash["cpu"]
            self._hash[key] = (*(t.to(device) for t in tensors), pad)
        return self._hash[key]

    def forward(self, input_ids, seq=None):
        from .piecewise import run
        return run(self, self._forward_impl, (input_ids,), {"seq": seq})

    def _forward_impl(self, input_ids, seq=None):
        cfg = self.cfg
        if self.hashing and self.placement is not None and self.placement.side == "engine" and seq is None:
            # Without the step's sequence context the hash would treat the packed batch as one sequence (n-grams
            # across requests and chunks): never silently.
            raise ValueError("ENGRAM_NO_SEQUENCE_CONTEXT: an engine forward of the Engram hash needs the step's "
                             "sequence context (seq); none was passed")
        lead = tuple(input_ids.shape)
        ids = input_ids.to(self.weight.device)
        if torch.is_grad_enabled() and self.weight.requires_grad:
            x = nn.functional.embedding(ids, self.weight)
            x = x.unsqueeze(-2).expand(*lead, cfg.hc_mult, cfg.hidden_size).contiguous()
            pre = torch.zeros(*lead, cfg.hc_mult, dtype=torch.float32, device=x.device)
            pre[..., 0] = 1
        else:
            from isoexec.ops.fused_sm100 import ops as fused
            x, pre = fused.embed_expand(self.weight, ids, cfg.hc_mult, one_hot_pre=True)
        if not self.hashing:
            return x, pre
        from isoexec.ops.embed.engram_sm100.ops import ngram_hash
        token_map, primes, offsets, multipliers, pad = self._hash_tensors(ids.device)
        if self.placement is not None and self.placement.side == "trainer":
            return x, pre, self._packed_hashes(ids, seq)
        engine = _engine_history(seq)
        if engine is None:
            if self.placement is not None and self.placement.side == "engine" and _has_step_metadata():
                raise ValueError("ENGRAM_NO_HISTORY: an engine step with attention metadata has no per-request "
                                 "n-gram history (the model state's lookback_token_ids); the whole-batch hash is "
                                 "only for vLLM's metadata-free profiling run")
            return x, pre, ngram_hash(ids, token_map, primes, offsets, multipliers, pad)
        return x, pre, self._engine_hashes(ids, *engine)

    def _packed_hashes(self, ids, seq):
        """Trainer: each packed sequence hashed from an empty history (no n-gram crosses a boundary), the
        values of its own row of the validated trainer's padded batch."""
        from isoexec.ops.embed.engram_sm100.ops import ngram_hash
        token_map, primes, offsets, multipliers, pad = self._hash_tensors(ids.device)
        bounds = None if seq is None or seq.cu_seqlens_cpu is None else seq.cu_seqlens_cpu.tolist()
        if bounds is None or len(bounds) == 2 or ids.shape[0] != 1:
            return ngram_hash(ids, token_map, primes, offsets, multipliers, pad)
        if bounds[-1] != ids.shape[1]:
            raise ValueError(f"packed sequence boundaries {bounds[-1]} do not cover {ids.shape[1]} tokens")
        # every sequence from its own empty history in ONE launch (the kernel's per-token sequence starts): the values
        # of one call per sequence, with one error-word check instead of one per sequence
        starts = torch.tensor([a for a, b in zip(bounds[:-1], bounds[1:]) for _ in range(b - a)], dtype=torch.int64)
        return ngram_hash(ids, token_map, primes, offsets, multipliers, pad,
                          seg_start=starts.to(ids.device, non_blocking=True))

    def _engine_hashes(self, ids, lookback, starts):
        """Packed engine tokens ``ids [1, N]``: each request hashed with its own history, the ids of the
        tokens before its chunk (``lookback [requests, N - 1]``, newest first, -1 none; vLLM's V4.1 model
        state) mapped to compressed ids, oldest first. ``starts``: host request boundaries, or None for
        single-token rows (captured decode: one launch over every row). Integer-exact; rows independent."""
        from isoexec.ops.embed.engram_sm100.ops import ngram_hash
        token_map, primes, offsets, multipliers, pad = self._hash_tensors(ids.device)
        N = ids.shape[1]
        if lookback.dtype == torch.int32 and lookback.is_contiguous():
            # the kernel maps the raw lookback itself (newest first, -1 none): the same compressed history ids as
            # the flip / token_map / where glue below, without its launches
            raw, history = lookback, None
        else:
            raw = None
            history = lookback.flip(-1).long()
            history = torch.where(history >= 0, token_map[history.clamp(min=0)], -1)

        def rows(lo, hi):
            return (dict(raw_state=raw[lo:hi]) if raw is not None else dict(state=history[lo:hi].contiguous()))
        if starts is None:
            out = ngram_hash(ids.view(N, 1), token_map, primes, offsets, multipliers, pad, validate=False,
                             **rows(0, N))
            return out.view(1, N, *out.shape[2:])
        parts, end = [], 0
        for s, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            if b > a:
                parts.append(ngram_hash(ids[:, a:b], token_map, primes, offsets, multipliers, pad, validate=False,
                                        **rows(s, s + 1)))
            end = b
        L, cols = primes.shape[0], primes.shape[1] * primes.shape[2]
        if end < N:
            parts.append(torch.zeros((1, N - end, L, cols), dtype=torch.int64, device=ids.device))
        return torch.cat(parts, dim=1)

    def abstract_output(self, input_ids, seq=None):
        cfg = self.cfg
        B, T = input_ids.shape
        dev = input_ids.device
        x = torch.empty((B, T, cfg.hc_mult, cfg.hidden_size), dtype=torch.bfloat16, device=dev)
        pre = torch.empty((B, T, cfg.hc_mult), dtype=torch.float32, device=dev)
        if not self.hashing:
            return x, pre
        L, cols = len(cfg.released_engram_layers), (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        return x, pre, torch.empty((B, T, L, cols), dtype=torch.int64, device=dev)


def _has_step_metadata() -> bool:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except ImportError:
        return False
    return is_forward_context_available() and get_forward_context().attn_metadata is not None


def _engine_history(seq):
    """``(lookback, host request starts | None)`` of an engine step with paged metadata, else None."""
    if seq is None:
        return None
    from isoexec.runtimes.vllm._inputs import current_engine_inputs
    inputs = current_engine_inputs()
    if inputs is None:
        return None
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available() or get_forward_context().attn_metadata is None:
        return None
    lookback = inputs.get("lookback_token_ids")
    if lookback is None:
        raise RuntimeError("Engram hashing on the engine needs the model state's lookback_token_ids")
    context = get_forward_context()
    if lookback.is_cuda and torch.cuda.is_current_stream_capturing():
        return lookback, None          # vLLM captures single-token decode batches only
    starts = next((v.query_start_loc_cpu for v in context.attn_metadata.values()
                   if getattr(v, "query_start_loc_cpu", None) is not None), None)
    if starts is None:
        raise RuntimeError("Engram hashing needs host request boundaries")
    starts = [int(v) for v in starts.tolist()]
    if all(b - a == 1 for a, b in zip(starts[:-1], starts[1:])) and starts[-1] == seq.positions.shape[-1]:
        return lookback, None
    return lookback, starts


def compressed_token_map(tokenizer) -> list[int]:
    """The compressed id of every tokenizer id: the checkpoint's pinned NFKC / NFD / accent / lowercase /
    whitespace normalization of each token's text (tokens that do not decode to text keep their own key)."""
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    keys, result = {}, []
    backend = tokenizer.backend_tokenizer
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "\ufffd" in text else normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        result.append(keys[key])
    return result


__all__ = ["DeepSeekV41Embed", "compressed_token_map"]
