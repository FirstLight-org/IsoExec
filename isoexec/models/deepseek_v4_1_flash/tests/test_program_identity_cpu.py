"""The forward program's identity travels with every weight update; the engine refuses a different trainer program."""

import pytest
import torch
from torch import nn

from isoexec.models.deepseek_v4_1_flash import config as v41_config
from isoexec.models.deepseek_v4_1_flash.config import BOUNDED_REPLAY, FULL_DECODER, ced_geometry, program_identity
from isoexec.models.deepseek_v4_1_flash.tests.test_ced_definition_cpu import SOURCE, raw
from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec
from isoexec.sync import identity


@pytest.fixture(scope="module")
def cfg():
    return ced_geometry(raw("config.json"))


def test_identity_names_the_index_score_program(cfg, monkeypatch):
    dg = program_identity(cfg, BOUNDED_REPLAY)
    assert dg["index_scores"] == "sm100_dsa_index_logits_dg_v1" == dg["attention_forward"]["indexer"]
    monkeypatch.setattr(spec, "INDEX_FOLD", "fp32_tree")
    tree = program_identity(cfg, BOUNDED_REPLAY)
    assert tree["index_scores"] == "sm100_dsa_indexer_tree_v1"
    diff = identity.differences(identity._plain(dg), identity._plain(tree))
    assert [d.split(":")[0] for d in diff] == ["attention_forward.indexer", "index_scores"]


def test_backward_provider_and_transport_fallbacks_are_recorded_not_compared(cfg, monkeypatch):
    base = program_identity(cfg, BOUNDED_REPLAY)
    recorded = v41_config.program_recorded()
    assert recorded == {"tp_reducer": "tree", "ep_exchange": "mega", "attention_backward": "cudnn_det",
                        "trained_set": "actor"}
    monkeypatch.setenv("ISOEXEC_DSA_BWD", "isoexec")
    monkeypatch.setenv("ISOEXEC_DS_FREEZE_ROUTED_EXPERTS", "1")
    monkeypatch.setenv("ISOEXEC_DS_TP_REDUCER", "all_gather")
    monkeypatch.setenv("ISOEXEC_DS_EP_EXCHANGE", "fixed_order")
    monkeypatch.setitem(spec.SUB_RECIPES, "sparse_bwd", "another_backward")
    assert program_identity(cfg, BOUNDED_REPLAY) == base and "sparse_bwd" not in base["attention_forward"]
    assert v41_config.program_recorded() == {"tp_reducer": "all_gather", "ep_exchange": "fixed_order",
                                             "attention_backward": "isoexec",
                                             "trained_set": "actor_routed_experts_frozen"}
    engine = {"identity": identity._plain(base), "recorded": recorded}
    identity.check(engine, identity.decode(identity.encode({"identity": base, "recorded":
                                                            v41_config.program_recorded()})))


def test_every_member_states_the_models_identity():
    model = SOURCE.build_meta(raw("b14_bf16_config.json"))
    stated = identity.stage_identity(model)
    assert stated["identity"] == identity._plain(model.program_identity())
    assert all(identity._plain(m.program_identity()) == stated["identity"] for m in model.members())
    # a pipeline stage holding only encoder blocks states the same identity
    stage = nn.Module()
    stage.blocks = nn.ModuleList(list(model.encoder.blocks)[:2])
    assert identity.stage_identity(stage) == stated


def test_payload_survives_a_bf16_stream(cfg):
    payload = {"identity": program_identity(cfg, BOUNDED_REPLAY), "recorded": v41_config.program_recorded()}
    tensor = identity.encode(payload)
    assert tensor.dtype == torch.bfloat16 and tensor.dim() == 1
    assert identity.decode(tensor.to(torch.bfloat16).clone()) == identity._plain(payload)
    with pytest.raises(ValueError):
        identity.decode(tensor + 0.5)


class _Member(nn.Module):
    def __init__(self, ident):
        super().__init__()
        self.ident = ident
        self.weight = nn.Parameter(torch.zeros(3))

    def program_identity(self):
        return self.ident

    def program_recorded(self):
        return {"tp_reducer": "tree"}


def _engine(ident):
    from isoexec.runtimes.vllm.model import VllmModel

    model = VllmModel.__new__(VllmModel)
    nn.Module.__init__(model)
    model.body = nn.Module()
    model.body.member = _Member(ident)
    model.head = None
    model._loaded_weights, model._weight_update_failed = None, False
    model._verify_weight_updates = False
    model._remote_updates, model._update_aliases = set(), {}
    model._copy_parameter_update = lambda name, parameter, tensor, loaded: loaded.add(parameter)
    return model


def _trainer_stream(ident):
    from types import SimpleNamespace

    from isoexec.integrations.skyrl.weight_source import IsoExecWeightSource

    module = nn.Module()
    module.member = _Member(ident)
    one = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    src = IsoExecWeightSource.__new__(IsoExecWeightSource)
    src._module, src._groups, src._tp_group, src._pp_group, src._head_name = module, None, one, one, None
    src._identity = identity.encode(identity.stage_identity(module))
    stream = list(src.iter_full())
    assert [m.name for m in src.metadata] == [identity.NAME, "body.member.weight"]
    assert stream[0][0] == identity.NAME
    return stream


def test_engine_checks_the_trainer_identity_at_every_update(cfg, monkeypatch):
    from isoexec.runtimes.vllm import model as vllm_model

    for name in ("refresh_packed_storage", "refresh_derived_weights"):
        monkeypatch.setattr(vllm_model, name, lambda body: None)
    monkeypatch.setattr(vllm_model.VllmModel, "derived_weight_check",
                        lambda self, rebuild=False: type("C", (), {"stale": lambda s: []})(), raising=False)
    monkeypatch.setattr("isoexec.runtimes.vllm.weights_sleep.mark_restored", lambda model: None)
    ident = identity._plain(program_identity(cfg, BOUNDED_REPLAY))
    engine = _engine(ident)
    for _ in range(2):                              # every update carries and checks it
        engine.start_weight_update()
        engine.load_weights(_trainer_stream(ident))
        engine.finish_weight_update()
    # a trainer on the old index-score tree
    monkeypatch.setattr(spec, "INDEX_FOLD", "fp32_tree")
    # The refusal is deferred to finish_weight_update: the stream is drained (a broadcast transport completes, the
    # trainer is not left waiting) and no Parameter is written; the finish reports it by name.
    before = {n: p.detach().clone() for n, p in engine.named_parameters()}
    engine.start_weight_update()
    stream = _trainer_stream(identity._plain(program_identity(cfg, BOUNDED_REPLAY)))
    consumed = []
    engine.load_weights((consumed.append(name) or (name, t)) for name, t in stream)
    assert consumed == [name for name, _ in stream]                         # drained to the end
    assert all(torch.equal(p, before[n]) for n, p in engine.named_parameters())   # nothing written
    with pytest.raises(ValueError, match=r"PROGRAM_IDENTITY_MISMATCH.*index_scores: engine='sm100_dsa_index_logits_dg_v1'"
                                         r" trainer='sm100_dsa_indexer_tree_v1'"):
        engine.finish_weight_update()
    with pytest.raises(ValueError, match="Weight update failed"):            # stays failed until a new start
        engine.finish_weight_update()
    monkeypatch.setattr(spec, "INDEX_FOLD", "bf16_dg")
    # another CED program
    engine.start_weight_update()
    engine.load_weights(_trainer_stream(identity._plain(program_identity(cfg, FULL_DECODER))))
    with pytest.raises(ValueError, match="PROGRAM_IDENTITY_MISMATCH.*ced_prefill"):
        engine.finish_weight_update()
    # a correct update after a refused one succeeds
    engine.start_weight_update()
    engine.load_weights(_trainer_stream(ident))
    engine.finish_weight_update()
    # an update without the identity entry
    engine.start_weight_update()
    engine.load_weights(_trainer_stream(ident)[1:])
    with pytest.raises(ValueError, match="PROGRAM_IDENTITY_MISSING"):
        engine.finish_weight_update()
