"""CPU structure gate of the auxiliary-stream x projections (paged._x_projections_aux).

The compressor's [wkv | wgate] and the indexer's weights_proj read only the layer input; on CUDA (no grad,
N <= MULTI_STREAM_TOKENS) they are issued on an auxiliary stream and joined before first use. The values
are the same calls on the same operands: ``index_queries(weights=index_weights(...))`` is
``index_queries(...)``, and the helper declines on CPU / grad / large N / non-source layers.
"""
from types import SimpleNamespace as NS

import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged


def test_index_weights_is_index_queries_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "mm", lambda a, w, **kw: (calls.append(tuple(w.shape)), a @ w.float().t().to(a.dtype))[1])
    monkeypatch.setattr(ops, "row_op", lambda *a, **k: None)
    monkeypatch.setattr(ops.spec, "index_weight_scale", lambda cfg: 0.125)
    cfg = NS(index_n_heads=64, index_head_dim=128)
    idx = torch.nn.Module()
    idx.weights_proj = NS(weight=torch.randn(64, 32).to(torch.bfloat16))
    idx.wq_b = None
    x = torch.randn(3, 32).to(torch.bfloat16)
    lin = lambda m, q: torch.zeros((1, 3, 64 * 128))
    _, _, w_ref = ops.index_queries(idx, x, torch.zeros(1, 3, 8), cfg, None, None, lin, (1, 3, 32), (1, 3, 8))
    n_ref = len(calls)
    w_early = ops.index_weights(idx, x, cfg)
    _, _, w_new = ops.index_queries(idx, x, torch.zeros(1, 3, 8), cfg, None, None, lin, (1, 3, 32), (1, 3, 8),
                                    weights=w_early)
    assert w_new is w_early and torch.equal(w_ref, w_early)
    assert len(calls) == n_ref + 1               # the early call only; index_queries issued no weights mm


def test_aux_projections_gate(monkeypatch):
    cfg = NS(kv_source_layer_ids=[2], index_source_layer_ids=[2, 4])
    cpu = torch.device("cpu")
    x = torch.zeros(4, 8)
    monkeypatch.setattr(paged, "_aux_stream", lambda dev: object())
    assert paged._x_projections_aux(None, x, cfg, 2, 2, 4, cpu) is None          # CPU: never
    assert paged._x_projections_aux(None, x, cfg, 0, 0, 4, torch.device("cuda")) is None   # not a source layer
    assert paged._x_projections_aux(None, x, cfg, 2, 2, paged.MULTI_STREAM_TOKENS + 1, torch.device("cuda")) is None
    monkeypatch.setattr(paged, "_aux_stream", lambda dev: None)                     # capturing before creation
    assert paged._x_projections_aux(None, x, cfg, 2, 2, 4, torch.device("cuda")) is None


class _Log:
    """Fake CUDA streams: records every launch with the stream it was issued on, and the stream waits."""

    def __init__(self):
        self.events, self.current = [], "main"

    def stream(self, name):
        log = self

        class S:
            device = torch.device("cuda")

            def __init__(self):
                self.name = name

            def wait_stream(self, other):
                log.events.append(("wait", self.name, other.name))

        return S()


def test_aux_region_is_finite_compute_and_joined_before_return(monkeypatch):
    """MegaMoE / collective ordering: everything issued on the auxiliary stream is an mm (finite compute; no
    collective, no cross-rank wait), and the layer's stream waits for the auxiliary stream (``_Early.join``)
    before anything after the layer -- the TP reduction and the MoE run later on that stream."""
    log = _Log()
    streams = {"main": log.stream("main"), "aux": log.stream("aux")}

    class Ctx:
        def __init__(self, s):
            self.s, self.prev = s, None

        def __enter__(self):
            self.prev, log.current = log.current, self.s.name

        def __exit__(self, *a):
            log.current = self.prev

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: streams[log.current])
    monkeypatch.setattr(torch.cuda, "stream", lambda s: Ctx(s))
    monkeypatch.setattr(paged, "_aux_stream", lambda dev: streams["aux"])

    class T:
        def contiguous(self):
            return self

        def record_stream(self, s):
            log.events.append(("record", s.name))

    def mm(a, w, **kw):
        log.events.append(("launch:mm", log.current))
        return T()
    monkeypatch.setattr(ops, "mm", mm)
    monkeypatch.setattr(ops, "index_weights", lambda idx, x2d, cfg: mm(x2d, None))
    monkeypatch.setattr(ops, "_bf16_weight", lambda *a: None)

    class Cache:
        def get(self, name, sources, make):
            return None
    monkeypatch.setattr(ops, "_cache", lambda m: Cache())
    comp = NS(wkv=NS(weight=None), wgate=NS(weight=None))
    module = NS(compressor=comp, indexer=None)
    cfg = NS(kv_source_layer_ids=[2], index_source_layer_ids=[2])
    with torch.no_grad():
        early = paged._x_projections_aux(module, T(), cfg, 2, 2, 8, torch.device("cuda"))
    assert early is not None
    launches = [e for e in log.events if e[0].startswith("launch")]
    assert launches == [("launch:mm", "aux"), ("launch:mm", "aux")]               # only finite GEMMs on aux
    assert log.events[0] == ("wait", "aux", "main")                                # aux starts after main's work
    early.take("comp")
    early.take("weights")
    early.join()
    joins = [e for e in log.events if e == ("wait", "main", "aux")]
    assert len(joins) == 1                                                         # joined once into main


def test_attention_joins_aux_before_output_projection():
    """Source-order guard: in ``attention_paged_gen`` the auxiliary-stream join precedes the output projection
    (wo_a / wo_b) and the return, so no auxiliary-stream work can be pending when the caller issues the TP
    reduction or the MoE (MegaMoE) on the layer's stream."""
    import inspect
    src = inspect.getsource(paged.attention_paged_gen)
    start = src.index("early = _x_projections_aux(")
    join = src.index("early.join()")
    out = src.index("ops.output_projection(")
    assert start < join < out and src.count("early.join()") == 1
    side = src.index("main_stream.wait_stream(side)")                              # the wq_b side stream
    assert src.index("side.wait_stream(main_stream)") < side < out
    assert "yield" not in src[start:]                                              # no host yield in between


def test_window_kv_record_joins_before_the_pools_are_read():
    """Source-order guard: the wkv projection and its window-KV record may run on the auxiliary stream beside wq_a,
    and the layer stream waits for them (``kv_done``) before the q norm's consumers and the attention lists."""
    import inspect
    src = inspect.getsource(paged.attention_paged_gen)
    issue = src.index("kv_done.record(aux)")
    wait = src.index("wait_event(kv_done)")
    assert src.index("linear_fn(module.wq_a") < wait                       # wq_a + q norm overlap the record
    assert issue < wait < src.index("early.join()") < src.index("ops.output_projection(")
    assert src.count("paged_quant_write(0, kvr") == 2                      # the same record on either stream


def test_q_kv_post_reads_the_window_projection_after_the_join():
    """Source-order guard of the one-launch form: wkv may run on the auxiliary stream beside wq_a, and the layer
    stream waits for it before q_kv_post (q_norm + window-KV record) reads its rows."""
    import inspect
    src = inspect.getsource(paged.attention_paged_gen)
    fused = src.index("if fused_post:")
    separate = src.index("else:", src.index("extension().q_kv_post(", fused))
    wait = src.index("wait_event(kv_done)", fused)
    assert src.index("kv_done.record(aux)", fused) < src.index("linear_fn(module.wq_a", fused) < wait
    assert wait < src.index("extension().q_kv_post_rows(", fused) < src.index("extension().q_kv_post(", fused)
    assert src.index("extension().q_kv_post(", fused) < separate < src.index("early.join()")


def test_grouped_wo_a_provider_selection(monkeypatch):
    """The linear provider selects DeepGEMM's grouped wo_a (``grouped_deepgemm``); host tensors keep grouped_gemm."""
    seen = []
    monkeypatch.setattr(ops, "grouped_gemm", lambda x, w, G, M: seen.append("grouped") or torch.zeros(G * M, w.shape[1]))
    monkeypatch.setattr(ops, "extension", lambda: NS(permute_rows=lambda y, out, G, M, f: None))
    og, wa = torch.zeros(2, 3, 4, dtype=torch.bfloat16), torch.zeros(2 * 5, 4, dtype=torch.bfloat16)
    ops._grouped_out(og, wa, 2, 5, deepgemm=True)                       # CPU operands: not DeepGEMM's
    assert seen == ["grouped"]
    assert not ops._dg_grouped_ok(og, wa)
