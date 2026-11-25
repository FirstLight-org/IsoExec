"""Resolve the same kernel environment before each fresh framework actor imports."""
import os


def resolved_environment(model_path: str) -> dict[str, str]:
    """The environment a fresh trainer or engine actor of an IsoExec job starts with (engines run on
    vLLM Model Runner V2)."""
    from isoexec.core.flags import FLAGS
    from isoexec.runtimes.trainer_installs import TRAINER_ENV

    env = dict(TRAINER_ENV)
    env.update({flag.name: os.environ[flag.name] for flag in FLAGS if flag.name in os.environ})
    # Fresh Ray workers inherit the cluster daemon's environment, which may
    # resolve a different cuDNN or write generated state into a source checkout.
    for name in (
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_LIB_PATH", "HF_HOME", "XDG_CACHE_HOME",
        "UV_CACHE_DIR", "TMPDIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_HOME", "PYTHONPYCACHEPREFIX",
        "PIK_CACHE", "FLASHINFER_WORKSPACE_BASE", "SGLANG_CACHE_DIR", "NVTE_FLASH_ATTN",
        "CPATH", "MAX_JOBS", "ISOEXEC_MOE_SM90_BUILD_DIR", "ISOEXEC_MM_SM90_BUILD_DIR", "WANDB_DIR",
        # placed by ISOEXEC_PREBUILT_DIR / ISOEXEC_PREBUILT_ONLY (ops/prebuilt.activate), which the FLAGS carry
        "TRITON_CACHE_MANAGER", "ISOEXEC_DG_JIT_NVCC_COMPILER", "CUTE_DSL_CACHE_DIR",
    ):
        if name in os.environ:
            env[name] = os.environ[name]
    for name in ("NCCL_ALGO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"):
        env.pop(name, None)
    env.update(
        ISOEXEC_MODEL_PATH=model_path,
        VLLM_PLUGINS="isoexec",
        # The fresh actor/child process has not called the driver's register().
        # Carry the explicit opt-in required by the installed plugin entrypoint.
        VLLM_ISOEXEC_REGISTERED="1",
        VLLM_USE_AOT_COMPILE="0",
        # The engine actor builds its VllmConfig before any in-process pin can run.
        VLLM_USE_V2_MODEL_RUNNER="1",
    )
    return env
