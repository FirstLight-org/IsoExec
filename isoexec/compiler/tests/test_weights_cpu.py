"""Checkpoint location resolution for model sources."""

import subprocess
import sys

import torch
from safetensors.torch import save_file


def test_local_checkpoint_resolution_does_not_import_transformers(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros(1)}, str(checkpoint))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from isoexec.compiler.weights import resolve_checkpoint\n"
            "assert str(resolve_checkpoint(sys.argv[1])) == sys.argv[1]\n"
            "assert 'transformers' not in sys.modules\n",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
