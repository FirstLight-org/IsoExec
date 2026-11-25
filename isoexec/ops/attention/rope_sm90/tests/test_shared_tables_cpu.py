"""Every RoPE closure in a process reads one table per constants, not one per layer."""

from types import SimpleNamespace

from isoexec.ops.attention.rope_sm90 import integration


def test_layers_share_one_table(monkeypatch):
    built = []

    def build_table(theta, rotary_dim, max_positions, device, **kw):
        built.append((theta, rotary_dim, max_positions, device))
        return SimpleNamespace(table=object())

    from isoexec.ops.attention.rope_sm90 import ops

    monkeypatch.setattr(ops, "build_table", build_table)
    monkeypatch.setattr(integration, "_SHARED_TABLES", {})
    sets = [integration._TableSet() for _ in range(10)]
    tables = {id(s.get("cuda:0", 1e7, 64)) for s in sets}
    assert len(built) == 1 and len(tables) == 1
    sets[0].get("cuda:0", 1e6, 64)
    assert len(built) == 2
