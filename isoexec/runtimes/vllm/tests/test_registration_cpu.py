"""Check native model registration with vLLM's registry. Registration and plugin installation must
not initialize CUDA before vLLM selects a worker device.
"""

import subprocess
import sys

import pytest

from isoexec.runtimes.vllm.registration import NATIVE_ARCH, NATIVE_DENSE_ARCH, register_native_model_class

vllm = pytest.importorskip("vllm", reason="the registry assertions need the pinned vLLM wheel")


def test_registry_holds_lazy_entries_for_both_native_classes():
    """String entries: resolving the class must not import the factory (and CUDA) at registration."""
    from vllm.model_executor.models.registry import ModelRegistry, _LazyRegisteredModel

    register_native_model_class()
    for arch, cls in ((NATIVE_ARCH, "VllmModel"), (NATIVE_DENSE_ARCH, "DenseVllmModel")):
        entry = ModelRegistry.models[arch]
        assert isinstance(entry, _LazyRegisteredModel), entry
        assert (entry.module_name, entry.class_name) == ("isoexec.runtimes.vllm.factory", cls)


def test_only_the_hybrid_arch_is_mapped_onto_vllms_hybrid_config_pass():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig

    register_native_model_class()
    assert MODELS_CONFIG_MAP[NATIVE_ARCH] is HybridAttentionMambaModelConfig
    assert NATIVE_DENSE_ARCH not in MODELS_CONFIG_MAP


def test_registration_is_idempotent_and_stable_across_repeats():
    """It runs once per process -- driver, EngineCore, every worker -- so repeats must not drift."""
    from vllm.model_executor.models.registry import ModelRegistry

    register_native_model_class()
    first = {arch: ModelRegistry.models[arch] for arch in (NATIVE_ARCH, NATIVE_DENSE_ARCH)}
    register_native_model_class()
    for arch, entry in first.items():
        again = ModelRegistry.models[arch]
        assert (again.module_name, again.class_name) == (entry.module_name, entry.class_name)


def test_importing_the_register_module_and_the_plugin_module_initializes_no_cuda():
    probe = (
        "import torch\n"
        "import isoexec.runtimes.vllm.registration\n"
        "import isoexec.runtimes.vllm.vllm_plugin\n"
        "print('CUDA_INIT:' + str(torch.cuda.is_initialized()))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "CUDA_INIT:False" in out.stdout, out.stdout[-2000:]


def test_register_plugin_initializes_no_cuda():
    """Run the real plugin entry point in a subprocess with production flags, including GDN, and
    require no CUDA context initialization and no factory import.
    """
    probe = (
        "import os, sys, torch\n"
        "from isoexec.runtimes.trainer_installs import TRAINER_ENV\n"
        "for k, v in TRAINER_ENV.items(): os.environ.setdefault(k, v)\n"
        "os.environ['VLLM_ISOEXEC_REGISTERED'] = '1'\n"
        "assert os.environ.get('ISOEXEC_GDN') == '1', os.environ.get('ISOEXEC_GDN')\n"
        "from isoexec.runtimes.vllm import vllm_plugin\n"
        "vllm_plugin.register()\n"
        "assert vllm_plugin._installed\n"
        "print('FACTORY:' + str('isoexec.runtimes.vllm.factory' in sys.modules))\n"
        "print('CUDA_INIT:' + str(torch.cuda.is_initialized()))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    assert "CUDA_INIT:False" in out.stdout, out.stdout[-3000:]
    assert "FACTORY:False" in out.stdout, out.stdout[-3000:]


def test_detect_arch_does_not_initialize_cuda_and_agrees_with_torch():
    """The NVML read must answer the same thing ``get_device_capability()`` would -- it is folded into
    the manifest hash, so a disagreement would silently re-key every bitwise gate table. Skipped
    where there is no accelerator; the comparison is the point, not the tag."""
    probe = (
        "import torch\n"
        "from isoexec.core.arch import NON_ACCELERATOR_ARCH, _arch_tag, detect_arch\n"
        "got = detect_arch()\n"
        "print('CUDA_INIT:' + str(torch.cuda.is_initialized()))\n"
        "if got == NON_ACCELERATOR_ARCH:\n"
        "    print('NO_ACCELERATOR')\n"
        "else:\n"
        "    print('AGREE:' + str(got == _arch_tag(*torch.cuda.get_device_capability())) + ':' + got)\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "CUDA_INIT:False" in out.stdout, out.stdout[-2000:]
    if "NO_ACCELERATOR" not in out.stdout:
        assert "AGREE:True:" in out.stdout, out.stdout[-2000:]
