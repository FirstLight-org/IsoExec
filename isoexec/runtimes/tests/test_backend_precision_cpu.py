"""Native worker precision setup versus real export, without CUDA execution."""

import pytest
import torch
from torch import nn
from typing import cast

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.nn import place
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.runtimes.precision import pin_fp32_ieee, _FP32PrecisionControl, _CuDNNPrecisionControl


def test_native_tf32_setup_requires_coherent_ieee_policy_before_export(monkeypatch):
    backends = cast(_FP32PrecisionControl, torch.backends)
    cudnn = cast(_CuDNNPrecisionControl, torch.backends.cudnn)
    previous = (cudnn.allow_tf32, backends.fp32_precision,
                torch.backends.cuda.matmul.fp32_precision, cudnn.fp32_precision,
                cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision)
    reductions = (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    def native_setup():
        # Public operations used by vLLM 0.28 Worker.__init__ and
        # batch_invariant.init_batch_invariance; no native worker is fabricated.
        cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        cudnn.conv.fp32_precision = "ieee"
        cudnn.rnn.fp32_precision = "ieee"

    class Add(nn.Module):
        def forward(self, value):
            return value + 1

    try:
        native_setup()
        with pytest.raises(RuntimeError, match="TF32"):
            torch.export.export(Add(), (torch.ones(4),), strict=False)
        pin_fp32_ieee()
        assert cudnn.allow_tf32 is False
        assert (backends.fp32_precision, torch.backends.cuda.matmul.fp32_precision,
                cudnn.fp32_precision, cudnn.conv.fp32_precision,
                cudnn.rnn.fp32_precision) == ("ieee",) * 5
        model = Model(device="cpu").eval()
        inputs = example_inputs()
        graph = capture(model, inputs)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.02)
        place(model, trivial_placement(), op_table=OpTable(mode="reference"))
        expected, actual = model(*inputs), graph.program(*inputs)
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        from isoexec.runtimes import trainer_installs

        installed = {"already_installed": True, "deterministic_attention_backward": False,
                     "attention_recipe": {"recipe": "FA3", "forward": "FA3", "backward": "FA3",
                                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                          "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
        monkeypatch.setattr(trainer_installs, "_INSTALLED", installed.copy())
        native_setup()
        assert trainer_installs.install_trainer_kernels() == installed
        assert cudnn.allow_tf32 is False
        assert (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction) == reductions
        assert not torch.cuda.is_initialized()
    finally:
        cudnn.allow_tf32 = previous[0]
        backends.fp32_precision = previous[1]
        torch.backends.cuda.matmul.fp32_precision = previous[2]
        cudnn.fp32_precision = previous[3]
        cudnn.conv.fp32_precision = previous[4]
        cudnn.rnn.fp32_precision = previous[5]
