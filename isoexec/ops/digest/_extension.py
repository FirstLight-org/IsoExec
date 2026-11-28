"""Build the multi-tensor digest extension lazily with the repo's arch build profile (no build on import)."""

from functools import lru_cache

SOURCES = ("digest.cu",)


@lru_cache(None)
def extension(arch: str | None = None):
    """The loaded ``isoexec_digest_<arch>_cuda`` module for ``arch`` (default: the live device); compiled on first
    call. A prebuild on a host without a device names the arch (``"sm100"``)."""
    from isoexec.core.arch import live_device_arch
    from isoexec.ops.build_cache import build_sm90_extension, build_sm100_extension

    arch = arch or live_device_arch()
    if arch == "sm90":
        return build_sm90_extension("digest", __file__, SOURCES)
    if arch == "sm100":
        return build_sm100_extension("digest", __file__, SOURCES)
    raise RuntimeError(f"digest: no device build for {arch}")
