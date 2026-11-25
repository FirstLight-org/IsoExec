"""FLA backward kernels with measured pins never autotune on that architecture."""

from types import SimpleNamespace

import pytest

from isoexec.ops.gdn.shared import gdn_batch_invariant as gbi


def _config(w, s):
    return SimpleNamespace(num_warps=w, num_stages=s, kwargs={})


def _tree(configs):
    autotuner = SimpleNamespace(configs=list(configs), cache={"stale": True})
    module = SimpleNamespace(chunk_bwd_kernel_dqkwg=SimpleNamespace(fn=autotuner))
    return autotuner, (lambda name: module)


def test_sm90_pins_every_measured_kernel():
    table = gbi._FLA_BACKWARD_PINS["sm90"]
    tuners, modules = {}, {}
    for module_name, kernel_name in table:
        configs = [SimpleNamespace(num_warps=w, num_stages=s, kwargs={"BV": bv})
                   for w in (1, 2, 4, 8) for s in (2, 3, 4) for bv in (32, 64)]
        tuners[kernel_name] = SimpleNamespace(configs=configs, cache={"stale": True})
        modules.setdefault(module_name, SimpleNamespace())
        setattr(modules[module_name], kernel_name, SimpleNamespace(fn=tuners[kernel_name]))
    everything = {f"{m}.{k}": tuners[k] for m, k in table}
    pinned = gbi.pin_fla_backward_configs(lambda name: modules[name], arch="sm90", all_autotuners=lambda: everything)
    assert len(pinned) == len(table)
    for (_, kernel_name), want in table.items():
        (chosen,) = tuners[kernel_name].configs
        assert tuners[kernel_name].cache == {}
        assert chosen.num_warps == want["num_warps"] and chosen.num_stages == want["num_stages"]
        assert chosen.kwargs["BV"] == want.get("BV", 32)


def test_unmeasured_arch_keeps_autotune(capsys):
    candidates = [_config(2, 2), _config(4, 3)]
    autotuner, importer = _tree(candidates)
    assert gbi.pin_fla_backward_configs(importer, arch="sm80") == []
    assert autotuner.configs == candidates
    assert "UNPINNED" in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_the_whole_tree_is_pinned(arch):
    """Measured kernels take their pin; every other autotuner its first candidate."""
    assert arch in gbi._FLA_BACKWARD_PIN_ALL
    table = gbi._FLA_BACKWARD_PINS[arch]
    modules, tuners = {}, {}
    for module_name, kernel_name in table:
        configs = [SimpleNamespace(num_warps=w, num_stages=s, kwargs={"BV": bv, "BK": bk})
                   for w in (1, 2, 4, 8) for s in (2, 3, 4) for bv in (32, 64, 128) for bk in (32, 64, 128)]
        tuners[kernel_name] = SimpleNamespace(configs=configs, cache={"stale": True})
        modules.setdefault(module_name, SimpleNamespace())
        setattr(modules[module_name], kernel_name, SimpleNamespace(fn=tuners[kernel_name]))
    other = SimpleNamespace(configs=[_config(8, 2), _config(1, 1)], cache={"stale": True})
    everything = {f"{m}.{k}": tuners[k] for m, k in table} | {"utils.other.other_kernel": other}
    gbi.pin_fla_backward_configs(lambda name: modules[name], arch=arch, all_autotuners=lambda: everything)
    assert all(len(t.configs) == 1 and t.cache == {} for t in everything.values())
    assert other.configs[0].num_warps == 8
    for (_, kernel_name), want in table.items():
        (chosen,) = tuners[kernel_name].configs
        assert all(getattr(chosen, k, chosen.kwargs.get(k)) == v for k, v in want.items())


def test_a_pin_that_matches_no_candidate_refuses():
    _, importer = _tree([_config(8, 2)])
    with pytest.raises(RuntimeError, match="matches the measured pin"):
        gbi.pin_fla_backward_configs(importer, arch="sm90")
