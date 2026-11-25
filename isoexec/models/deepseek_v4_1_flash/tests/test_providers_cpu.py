"""CPU contract of the actor recipe's provider seams: the TP reducer's dispatch and execution scope, the engine
MoE seam's MegaMoE selection and fallbacks, the DeepGEMM linear provider and the FP32 head's vocab shard check. The bits are GPU-gated by the ops' batteries (mm/deepgemm_sm100, moe/deepseek_v41_sm100)."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from isoexec.models.deepseek_v4_1_flash import providers
from isoexec.models.deepseek_v4_1_flash.providers import ActorTPReducer


class _R:
    world = 2

    def __init__(self, tag):
        self.tag, self.entered, self.held = tag, [], False

    def gather_sum(self, x):
        return self.tag

    def holds_execution_scope(self):
        return self.held

    @contextmanager
    def execution_scope(self, *, capture=False):
        self.entered.append(capture)
        yield


def test_actor_tp_reducer_dispatch(monkeypatch):
    tree, gather = _R("tree"), _R("gather")
    tree.held = True
    red = ActorTPReducer(tree, gather)
    x = torch.zeros(2, dtype=torch.float32)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
    with torch.no_grad():
        assert red.gather_sum(x) == "tree" and red.gather_sum(x.bfloat16()) == "gather"
    with torch.enable_grad():
        assert red.gather_sum(x) == "tree"                              # untracked payloads (the trainer forward)
        assert red.gather_sum(x.clone().requires_grad_()) == "gather"   # tracked payloads: all_gather + tree
        tree.held = False
        with pytest.raises(providers.TreeScopeNotHeld):                 # outside the scope: refused by name, never
            red.gather_sum(x)                                           # a silent per-thread transport switch
    monkeypatch.undo()
    tree.held = True
    with torch.no_grad():
        assert red.gather_sum(x) == "gather"                            # host tensors: the all-gather form


def test_actor_tp_reducer_scope_is_entered_unless_held(monkeypatch):
    tree = _R("tree")
    red = ActorTPReducer(tree, _R("gather"))
    with red.scope():
        pass
    assert tree.entered == [False]
    tree.held = True                       # an owning runtime's scope (capture / replay): not re-entered
    with red.scope():
        pass
    assert tree.entered == [False]
    tree.held = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with red.scope():                      # a capture without a runtime scope: the all-gather form, no scope
        pass
    assert tree.entered == [False]


def _block(ep=2, partitioned=False, mega=None, static=False, static_rows=None):
    cfg = NS(num_experts_per_tok=6, routed_scaling_factor=1.5, swiglu_limit=10.0)
    comm = NS(static_routes=static)
    gu, dn = torch.zeros(3, 8, 4), torch.zeros(3, 4, 4)
    return NS(cfg=cfg, ffn=NS(gate=NS(weight=torch.zeros(4, 4), bias=torch.zeros(4))),
              packed_experts=lambda: (gu, dn),
              ep=NS(size=ep, rank=1, group=None, partitioned=partitioned, expert_map="emap", mega=mega,
                    static_home_rows=static_rows, comm=lambda: comm))


def test_engine_ep_seam_selects_megamoe(monkeypatch):
    from isoexec.ops.moe.deepseek_v41_sm100 import ep as ep_ops
    calls = []
    monkeypatch.setattr(providers, "_ep_moe", lambda block, rows: calls.append("exchange") or rows)
    monkeypatch.setattr(ep_ops, "mega_ep_forward", lambda x, *a, **kw: calls.append(("mega", x.shape[0], kw["width"],
                                                                                      kw.get("out_rows"))) or x)
    rows = torch.zeros(7, 4)
    with torch.no_grad():
        providers._engine_ep_moe(_block(mega=None), rows)                # no workspace: the fixed-order exchange
        providers._engine_ep_moe(_block(mega="ws"), rows)                # replicated: home slice + fused gather
        providers._engine_ep_moe(_block(mega="ws", partitioned=True, static=True, static_rows=5), rows[:4])
    assert calls == ["exchange", ("mega", 4, 4, 7), ("mega", 4, 5, None)]
    with torch.enable_grad():                                            # grad-enabled: never MegaMoE
        providers._engine_ep_moe(_block(mega="ws"), rows.clone().requires_grad_())
    assert calls[-1] == "exchange"
    with pytest.raises(ValueError, match="capacity"):
        providers._engine_ep_moe(_block(mega="ws", partitioned=True, static=True, static_rows=2), rows[:4])


def test_engine_local_seam_keeps_host_and_grad_on_grouped(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "_local_moe", lambda block, rows: calls.append("grouped") or rows)
    with torch.no_grad():
        providers._engine_local_moe(_block(ep=1), torch.zeros(3, 4))      # host rows
    assert calls == ["grouped"]


def test_linear_provider_on_host_is_f_linear():
    lin = providers.BF16Linear()
    x, w = torch.randn(3, 64, dtype=torch.bfloat16), torch.randn(32, 64, dtype=torch.bfloat16)
    assert torch.equal(lin(NS(weight=w), x), torch.nn.functional.linear(x, w))
    assert lin.grouped_deepgemm and not getattr(lin, "supports_leaf_tree", False)


def test_vocab_shard_check():
    cfg = NS(vocab_size=129280, hidden_size=5120)
    assert all(providers.vocab_shard_is_bit_neutral(cfg, tp) for tp in (1, 2, 4, 8))
    assert not providers.vocab_shard_is_bit_neutral(NS(vocab_size=129281, hidden_size=5120), 2)


def test_linear_provider_keeps_narrow_weights_at_many_rows_on_the_mm_chain(monkeypatch):
    from isoexec.ops.mm.deepgemm_sm100 import bf16
    seen = []
    monkeypatch.setattr(providers, "install_projection_gemm", lambda: None)
    monkeypatch.setattr(bf16, "linear", lambda x, w: seen.append("deepgemm") or torch.zeros(()))
    monkeypatch.setattr(torch, "mm", lambda a, b: seen.append("mm") or torch.zeros(a.shape[0], b.shape[1]))
    monkeypatch.setattr(torch.nn.functional, "linear", lambda x, w: seen.append("F.linear"))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
    lin = providers.BF16Linear()
    for n, rows in ((4096, 256), (4096, 257), (4096, 2048), (8192, 2048), (32768, 2048)):
        lin(NS(weight=torch.empty(n, 16)), torch.empty(rows, 16))
    assert seen == ["deepgemm", "mm", "mm", "deepgemm", "deepgemm"]
    seen.clear()
    for n in (1792, 1856):                          # the packed wq_a | wkv (| weights_proj): mm_sm100 at every M
        for rows in (1, 128, 8192):
            lin(NS(weight=torch.empty(n, 5120)), torch.empty(rows, 5120))
    assert seen == ["mm"] * 6


def test_actor_tp_reducer_names_its_transport_and_preparation():
    tree = _R("tree")
    tree.prepare = lambda device: None
    red = ActorTPReducer(tree, _R("gather"))
    assert not red.prepared and red.transport() == "all_gather"
    assert red.prepare("cuda") is red and red.prepared
    tree.held = True
    assert red.transport() == "tree"


def test_fallback_switches_force_the_nccl_forms_on_every_rank(monkeypatch):
    """ISOEXEC_DS_TP_REDUCER=all_gather / ISOEXEC_DS_EP_EXCHANGE=fixed_order: read once at construction, agreed by
    the group (one rank asking is enough), never a symmetric setup; unknown values are refused."""
    import torch.distributed as dist

    from isoexec.models.deepseek_v4_1_flash import block as block_mod
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    for env, fallback in ((providers.TP_REDUCER_ENV, "all_gather"), (providers.EP_EXCHANGE_ENV, "fixed_order")):
        monkeypatch.delenv(env, raising=False)
        assert not providers.fallback_forced(env)
        monkeypatch.setenv(env, fallback.upper())
        assert providers.fallback_forced(env)
        monkeypatch.setenv(env, "nvlink")
        with pytest.raises(ValueError, match=env):
            providers.fallback_forced(env)
        monkeypatch.delenv(env)
    from isoexec.ops.collectives import node_local
    monkeypatch.setattr(node_local, "symmetric_admission",
                        lambda group: node_local.PeerAdmission(True, "one_host_distinct_gpus", "stand-in"))
    for mine, peer, want_tree in ((False, False, True), (True, False, False), (False, True, False)):
        def gather(out, flag, group=None, peer=peer):
            out[:] = [flag, not peer]
        monkeypatch.setattr(dist, "all_gather_object", gather)
        if mine:
            monkeypatch.setenv(providers.TP_REDUCER_ENV, "all_gather")
        else:
            monkeypatch.delenv(providers.TP_REDUCER_ENV, raising=False)
        tree, prepared = _R("tree"), []
        tree.group, tree.prepare = object(), prepared.append
        red = ActorTPReducer(tree, _R("gather"))
        monkeypatch.delenv(providers.TP_REDUCER_ENV, raising=False)    # read at construction only
        red.prepare("cuda")
        assert (red.tree_local, red.prepared, bool(prepared)) == (want_tree, want_tree, want_tree)
        tree.held = True
        assert red.transport() == ("tree" if want_tree else "all_gather")
        assert list(providers.TRANSPORT_RECEIPTS["tp_reducer"].values())[-1].startswith("tree" if want_tree else "all_gather")
    # the engine MoE workspace: the switch on one rank keeps every rank on the fixed-order exchange
    monkeypatch.setenv(providers.EP_EXCHANGE_ENV, "fixed_order")
    seen = []
    monkeypatch.setattr(block_mod, "_group_agrees", lambda group, flag: seen.append(flag) or flag)
    row = NS(ep=NS(size=2, group=object()), placement=NS(side="engine"),
             packed_experts=lambda: (NS(is_cuda=True), None))
    assert block_mod.V41Layer.prepare_expert_workspace(row) is False and seen == [False]


def test_engine_moe_seam_records_the_expert_path(monkeypatch):
    from isoexec.ops.moe.deepseek_v41_sm100 import ep as ep_ops
    from isoexec.ops.moe.deepseek_v41_sm100 import ops as moe_ops
    monkeypatch.setattr(providers, "_local_moe", lambda block, rows: rows)
    monkeypatch.setattr(providers, "_ep_moe", lambda block, rows: rows)
    monkeypatch.setattr(moe_ops, "moe_forward_mega", lambda rows, *a, **k: rows)
    monkeypatch.setattr(ep_ops, "mega_ep_forward", lambda rows, *a, **k: rows)
    block = _block(ep=1)
    with torch.no_grad():
        providers._engine_local_moe(block, torch.zeros(3, 4))
        assert block.expert_path == "grouped"                           # host rows
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
        providers._engine_local_moe(block, torch.zeros(3, 4))
        assert block.expert_path == "mega"
        block = _block(ep=2)
        providers._engine_ep_moe(block, torch.zeros(4, 4))
        assert block.expert_path == "fixed_order"                       # no workspace
        block.ep.mega = object()
        providers._engine_ep_moe(block, torch.zeros(4, 4))
        assert block.expert_path == "mega"


def test_pack_rows_repoints_shared_input_weights_without_changing_values():
    from torch import nn
    a = nn.Parameter(torch.randn(6, 8).bfloat16())
    b = nn.Parameter(torch.randn(2, 8).bfloat16())
    va, vb = a.detach().clone(), b.detach().clone()
    assert providers.adjacent_rows([a, b]) is None
    w = providers.pack_rows([a, b])
    assert w.shape == (8, 8) and torch.equal(w[:6], va) and torch.equal(w[6:], vb)
    assert torch.equal(a, va) and torch.equal(b, vb)
    assert providers.pack_rows([a, b]).data_ptr() == w.data_ptr()       # idempotent: an alias, no copy
    with torch.no_grad():
        b.copy_(torch.ones_like(b))                                     # a weight update writes into the pack
    assert torch.equal(w[6:], torch.ones_like(vb))
    assert providers.adjacent_rows([b, a]) is None                      # order matters
    assert providers.pack_rows([a, nn.Parameter(torch.randn(2, 4).bfloat16())]) is None


def test_packed_linear_splits_one_projection_and_falls_back_per_member(monkeypatch):
    from torch import nn
    lin = providers.BF16Linear()
    a, b = NS(weight=nn.Parameter(torch.randn(6, 8).bfloat16())), NS(weight=nn.Parameter(torch.randn(2, 8).bfloat16()))
    x = torch.randn(3, 8).bfloat16()
    calls = []
    real = providers.BF16Linear.__call__
    monkeypatch.setattr(providers.BF16Linear, "__call__", lambda s, h, t: calls.append(h.weight.shape[0]) or real(s, h, t))
    with torch.no_grad():
        assert not lin.packs((a, b))
        sep = lin.packed((a, b), x)
        assert calls == [6, 2]
        providers.pack_rows([a.weight, b.weight])
        assert lin.packs((a, b))
        calls.clear()
        got = lin.packed((a, b), x)
        assert calls == [8] and [t.shape for t in got] == [(3, 6), (3, 2)]
        assert all(torch.equal(g, s) for g, s in zip(got, sep))
    calls.clear()
    with torch.enable_grad():
        lin.packed((a, b), x)                                           # tracked weights: one projection each
    assert calls == [6, 2]


def test_tree_scope_decides_the_transport_structurally(monkeypatch):
    """Inside the scope: the tree. A capture no scope covers (no owner at all): the all-gather form, as every rank
    captures alike. A reduction from a thread that does not own the scope while another thread does (dual-batch
    micro-batch threads, async helpers), or outside any scope: refused by name (``TreeScopeNotHeld``)."""
    tree, gather = _R("tree"), _R("gather")
    owner = []
    tree.execution_scope_owner = lambda: owner[0] if owner else None
    red = ActorTPReducer(tree, gather)
    x = torch.zeros(2, dtype=torch.float32)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    with torch.no_grad():
        tree.held = True
        assert red.gather_sum(x) == "tree"
        tree.held = False
        capturing[0] = True
        assert red.gather_sum(x) == "gather"                  # scope-free capture
        owner.append(12345)                                   # another thread owns the scope
        with pytest.raises(providers.TreeScopeNotHeld, match="thread 12345"):
            red.gather_sum(x)
        capturing[0] = False
        with pytest.raises(providers.TreeScopeNotHeld):
            red.gather_sum(x)
        owner.clear()
        with pytest.raises(providers.TreeScopeNotHeld, match="scope owner: none"):
            red.gather_sum(x)
        red.tree_local = False                                # the group agreed the all-gather form at prepare
        assert red.gather_sum(x) == "gather"


def test_tp_reducer_takes_the_all_gather_form_by_the_agreed_peer_reason(monkeypatch):
    from isoexec.ops.collectives import node_local
    from isoexec.models.deepseek_v4_1_flash import block as block_mod
    monkeypatch.setattr(block_mod, "_group_agrees", lambda group, flag: flag)
    for reason in ("shared_device", "multi_host", "overlapping_ordinals"):
        monkeypatch.setattr(node_local, "symmetric_admission",
                            lambda group, reason=reason: node_local.PeerAdmission(False, reason, "stand-in"))
        tree, prepared = _R("tree"), []
        tree.group, tree.prepare = object(), prepared.append
        red = ActorTPReducer(tree, _R("gather"))
        assert red.prepare("cuda") is red
        assert (red.tree_local, red.prepared, prepared) == (False, False, [])        # no symmetric setup at all
        assert providers.TRANSPORT_RECEIPTS["tp_reducer"][repr(tree.group)] == f"all_gather:{reason}"
        assert red.transport() == "all_gather" and red.gather_sum(torch.zeros(2)) == "gather"


def test_expert_workspace_keeps_the_fixed_order_exchange_by_the_agreed_peer_reason(monkeypatch):
    from isoexec.ops.collectives import node_local
    from isoexec.models.deepseek_v4_1_flash import block as block_mod
    monkeypatch.delenv(providers.EP_EXCHANGE_ENV, raising=False)
    monkeypatch.setattr(block_mod, "_group_agrees", lambda group, flag: flag)
    monkeypatch.setattr(node_local, "symmetric_admission",
                        lambda group: node_local.PeerAdmission(False, "shared_device", "ranks [0, 1] share one GPU"))
    monkeypatch.setattr(providers, "prepare_expert_workspace",
                        lambda *a: (_ for _ in ()).throw(AssertionError("workspace prepared")))
    group = object()
    row = NS(ep=NS(size=2, group=group), placement=NS(side="engine"),
             packed_experts=lambda: (NS(is_cuda=True, device="cuda"), None))
    assert block_mod.V41Layer.prepare_expert_workspace(row) is False
    assert providers.TRANSPORT_RECEIPTS["ep_exchange"][repr(group)] == "fixed_order:shared_device"

def test_padding_rows_get_no_routed_expert_and_real_rows_are_untouched(monkeypatch):
    """``mega.skip_rows``: flagged rows' first k ids -> -1, weights -> 0 (never dispatched), the shared-expert column
    and every other row unchanged; ``providers.padding_rows`` reads vLLM's ``forward_context.is_padding`` (None
    outside a forward or without flags)."""
    import torch
    from isoexec.models.deepseek_v4_1_flash import providers
    from isoexec.ops.moe.deepseek_v41_sm100 import mega
    k = 3
    ids = torch.tensor([[5, 1, 7, 9], [2, 3, 4, 9], [8, 0, 6, 9]], dtype=torch.int32)
    w = torch.tensor([[.5, .3, .2, 1.], [.4, .4, .2, 1.], [.6, .3, .1, 1.]])
    ids0, w0 = ids.clone(), w.clone()
    skip = torch.tensor([False, True, False, True])      # longer than the rows: sliced to them
    mega.skip_rows(ids, w, skip, k)
    assert torch.equal(ids[[0, 2]], ids0[[0, 2]]) and torch.equal(w[[0, 2]], w0[[0, 2]])
    assert ids[1, :k].tolist() == [-1, -1, -1] and w[1, :k].tolist() == [0., 0., 0.]
    assert ids[1, k] == 9 and w[1, k] == 1.
    mega.skip_rows(ids, w, None, k)                       # no flags: nothing changes
    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: False)
    assert providers.padding_rows(4) is None
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(fc, "get_forward_context", lambda: NS(is_padding=skip))
    assert torch.equal(providers.padding_rows(3), skip[:3])
    assert providers.padding_rows(5) is None              # flags shorter than the rows: no mask
    monkeypatch.setattr(fc, "get_forward_context", lambda: NS(is_padding=None))
    assert providers.padding_rows(3) is None
