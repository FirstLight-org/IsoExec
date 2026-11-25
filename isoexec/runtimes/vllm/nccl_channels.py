"""Channel bounds on vLLM's original communicators, without process-wide pins."""

from __future__ import annotations

import ctypes
import functools
import inspect
import os

from isoexec.checks import Refusal

# Verified NCCL versions share the seven-field prefix through maxCTAs.
# Widen this range only after checking the target release's nccl.h field order
# and ABI; a reordered field could silently change collective channel geometry.
NCCL_ABI_VERIFIED = (22602, 23007)
_UNDEFINED = -(1 << 31)
_PROCESS_PINS = ("NCCL_ALGO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
_CAP = None
_RECEIPTS = []


class _NcclConfig(ctypes.Structure):
    """The ncclConfig_t prefix through maxCTAs, with seven fields. Its size advertises only the
    initialized prefix, leaving later fields at NCCL defaults across supported ABIs.
    """

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int),
        ("cgaClusterSize", ctypes.c_int),
        ("minCTAs", ctypes.c_int),
        ("maxCTAs", ctypes.c_int),
    ]

    def __init__(self, max_ctas, runtime_version):
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.magic = 0xCAFE_BEEF
        #: The version the caller targets. It is the running library's own, because this struct is
        #: built at runtime against whatever is loaded, not compiled against one release's header.
        self.version = runtime_version
        self.blocking = _UNDEFINED
        self.cgaClusterSize = _UNDEFINED
        self.minCTAs = 1
        self.maxCTAs = max_ctas


def clear_process_pins(env=None):
    target = os.environ if env is None else env
    for name in _PROCESS_PINS:
        target.pop(name, None)


def _require_unpinned():
    present = {name: os.environ[name] for name in _PROCESS_PINS if os.environ.get(name)}
    if present:
        raise Refusal(f"nccl.process_environment: unexpected process-wide pins before communicator creation: {present}")


def _device_identity() -> str:
    """``cuda:<index> uuid=<GPU uuid> visible=<CUDA_VISIBLE_DEVICES>`` of the calling process, for the
    log line above. Never raises: it is a diagnostic, and the communicator call is what must decide."""
    try:
        import torch

        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return f"cuda:{index} uuid={getattr(props, 'uuid', '?')} visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    except Exception as error:  # noqa: BLE001 -- diagnostic only
        return f"unknown ({type(error).__name__}: {error})"


def _initialize_rank(library, world_size, unique_id, rank, cap):
    _require_unpinned()
    version = library.ncclGetRawVersion()
    low, high = NCCL_ABI_VERIFIED
    if not low <= version <= high:
        raise Refusal(
            f"nccl.cta_bounds: this build sets minCTAs/maxCTAs through ncclConfig_t, and the field "
            f"order is verified for NCCL {low}..{high}; got {version}. Read that release's nccl.h and "
            f"confirm the first seven fields (size, magic, version, blocking, cgaClusterSize, "
            f"minCTAs, maxCTAs) are unchanged, then widen NCCL_ABI_VERIFIED."
        )
    if ctypes.sizeof(unique_id) != 128:
        raise Refusal("nccl.unique_id: expected the pinned 128-byte ncclUniqueId")
    try:
        initialize = library.lib.ncclCommInitRankConfig
    except AttributeError as error:
        raise Refusal("nccl.ncclCommInitRankConfig: required symbol is missing") from error
    initialize.restype = ctypes.c_int
    initialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                          type(unique_id), ctypes.c_int, ctypes.POINTER(_NcclConfig)]
    config = _NcclConfig(cap, version)
    communicator = ctypes.c_void_p()
    # Report group membership before initialization so duplicate-device errors are diagnosable.
    print(f"[ISOEXEC-NCCL] InitRankConfig pid={os.getpid()} world={world_size} rank={rank} max_ctas={cap} "
          f"device={_device_identity()}", flush=True)
    library.NCCL_CHECK(initialize(ctypes.byref(communicator), world_size, unique_id, rank, ctypes.byref(config)))
    _RECEIPTS.append({"backend": "pynccl", "world_size": world_size, "rank": rank,
                      "min_ctas": config.minCTAs, "max_ctas": config.maxCTAs,
                      "nccl_version": version, "config_size": config.size})
    return communicator


def install_pynccl_channel_policy(cap=None):
    """Use InitRankConfig for the original PyNccl communicator, including transfer."""
    global _CAP
    if cap is None:
        from isoexec.ops.collectives.nccl_identity import engine_channel_cap

        resolved = engine_channel_cap()
        if resolved not in ("8", "16"):
            raise Refusal(f"ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS: expected 8 or 16, got {resolved!r}")
        cap = int(resolved)
    if cap not in (8, 16):
        raise Refusal(f"nccl.max_ctas: certified engine ceilings are 8 and 16, got {cap!r}")
    if _CAP is not None and _CAP != cap:
        raise Refusal(f"nccl.max_ctas: this process already installed {_CAP}, cannot replace it with {cap}")
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    clear_process_pins()
    original = NCCLLibrary.ncclCommInitRank
    if getattr(original, "_isoexec_channels", False):
        return

    @functools.wraps(original)
    def initialize(library, world_size, unique_id, rank):
        return _initialize_rank(library, world_size, unique_id, rank, cap)

    initialize._isoexec_channels = True
    NCCLLibrary.ncclCommInitRank = initialize
    _CAP = cap


def _wrap_torch_group(original, cap):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def create(*args, **kwargs):
        import torch
        import torch.distributed as dist

        bound = signature.bind_partial(*args, **kwargs)
        backend = bound.arguments.get("backend")
        if backend is None and dist.is_initialized():
            backend = dist.get_backend()
        is_nccl = backend is None or str(backend) == "nccl" or "cuda:nccl" in str(backend)
        if not is_nccl:
            return original(*args, **kwargs)
        _require_unpinned()
        options = bound.arguments.get("pg_options")
        if options is None:
            options = dist.ProcessGroupNCCL.Options()
        if not isinstance(options, dist.ProcessGroupNCCL.Options):
            raise Refusal("nccl.pg_options: expected ProcessGroupNCCL.Options")
        for field, wanted in (("min_ctas", 1), ("max_ctas", cap)):
            actual = int(getattr(options.config, field))
            if actual not in (_UNDEFINED, wanted):
                raise Refusal(f"nccl.pg_options.{field}: supplied {actual} conflicts with {wanted}")
            setattr(options.config, field, wanted)
        bound.arguments["pg_options"] = options
        result = original(*bound.args, **bound.kwargs)
        group = dist.distributed_c10d._get_default_group() if result is None else result
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            config = group._get_backend(torch.device("cuda")).options.config
            if (int(config.min_ctas), int(config.max_ctas)) != (1, cap):
                raise Refusal("nccl.pg_options: original communicator did not retain the requested channel bounds")
            _RECEIPTS.append({"backend": "torch", "world_size": dist.get_world_size(group),
                              "rank": dist.get_rank(group), "min_ctas": 1, "max_ctas": cap})
        return result

    create._isoexec_channels = True
    return create


def install_engine_channel_policy(cap=16):
    """Configure the original torch and PyNccl engine groups before creation."""
    import torch.distributed as dist

    install_pynccl_channel_policy(cap)
    from isoexec.ops.collectives.nccl_identity import declare_engine_channels

    declare_engine_channels(cap)
    for name in ("init_process_group", "new_group"):
        original = getattr(dist, name)
        if not getattr(original, "_isoexec_channels", False):
            setattr(dist, name, _wrap_torch_group(original, cap))
