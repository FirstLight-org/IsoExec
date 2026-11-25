"""ModelContract has one model-owned definition and one state compatibility alias."""

from isoexec.compiler.contract.contract import ModelContract
from isoexec.state.contract import ModelContract as StateModelContract


def test_model_contract_has_one_definition():
    assert StateModelContract is ModelContract
    assert ModelContract.__module__ == "isoexec.compiler.contract.contract"
