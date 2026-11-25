"""Native completion makes updated registered weights visible before export."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.integrations.skyrl.tests import test_native_weight_sync_cpu

framework = test_native_weight_sync_cpu.framework
worker = test_native_weight_sync_cpu.worker
qwen_stage = test_native_weight_sync_cpu.qwen_stage
production_bindings = test_native_weight_sync_cpu.production_bindings
tp_group = test_native_weight_sync_cpu.tp_group
vllm_config = test_native_weight_sync_cpu.vllm_config
pair = test_native_weight_sync_cpu.pair


@pytest.mark.parametrize("overlap", [False, True])
def test_optimizer_completion_exports_current_weights_and_preserves_ties(pair, framework, monkeypatch, overlap):
    from megatron.core.distributed import DistributedDataParallel

    policy, source, _ = pair
    parameter = policy.graph_module.embed.weight
    ddp = DistributedDataParallel(policy.config, SimpleNamespace(), policy, pg_collection=policy.pg_collection)
    def finish_gather(*, skip_next_bucket_dispatch):
        assert overlap and skip_next_bucket_dispatch
        with torch.no_grad():
            parameter.fill_(0.75)

    ddp.bucket_groups = [SimpleNamespace(ddp_config=SimpleNamespace(overlap_param_gather=overlap),
                                         finish_param_sync=finish_gather)]
    ddp.expert_parallel_bucket_groups = []
    ddp.start_param_sync = lambda **kwargs: pytest.fail("Completion must not restart parameter gathers")

    if not overlap:
        with torch.no_grad():
            parameter.fill_(0.75)
    model = SimpleNamespace(actor_module=[ddp])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    framework.integration.finish_optimizer_step(model)

    assert policy.graph_module.embed.weight is policy.graph_module.head.weight is parameter
    assert torch.equal(dict(source.iter_full())["body.embed.weight"], torch.full_like(parameter, 0.75))
