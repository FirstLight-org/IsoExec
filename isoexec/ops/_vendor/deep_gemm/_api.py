import atexit
import os
import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
from . import _C
from ._C import (
    shutdown,
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_ignore_compile_dims,
    set_block_size_multiple_of,
    set_leaf_register_partials,
    get_leaf_register_partials,
    set_pdl,
    get_pdl,
    set_early_launch,
    get_early_launch,
    use_deterministic_algorithms,
)

# DeepGEMM Kernels
# IsoExec (pruned to BF16): only the entry points the IsoExec BF16 model calls are vendored. The FP8/FP4
# GEMM, einsum and MegaMoE bindings, the other BF16 GEMM layouts and groupings, and the cuBLASLt, attention
# (MQA logits), hyperconnection, MegaMHC and scaling-factor layout bindings are not part of this copy (see
# PROVENANCE.md, "Pruned to BF16"). MegaGate is vendored only as IsoExec's router program
# (``bf16_mega_gate_isoexec``); upstream's ``bf16_mega_gate`` scoring is not part of this copy.
from ._C import (
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nt_leaves,
    m_grouped_bf16_gemm_nt_contiguous, m_grouped_bf16_gemm_nn_contiguous, k_grouped_bf16_gemm_tn_contiguous,
    # Einsum kernels (``bhr,hdr->bhd``)
    einsum,
    # MegaMoE
    get_block_m_for_mega_moe,
    # MegaGate (IsoExec router program)
    bf16_mega_gate_isoexec, get_mega_gate_isoexec_config,
)

# Mega kernels
from .mega import (
    SymmBuffer,
    bf16_mega_moe,
)

# Epilogue classes
from ._C import epilogue

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    is_localized,
    destroy_localizer,
)

# Some utils
from . import utils
from .utils import *

# Initialize CPP modules
# IsoExec: this copy's JIT is private (see _extension.py and PROVENANCE.md). Its cache-key
# namespace folds the pinned revision and the include-tree digest, its environment prefix is
# ISOEXEC_DG (DG_* configures upstream copies only) and its disk cache is its own directory.
from . import UPSTREAM_REVISION as _UPSTREAM_REVISION
from ._extension import include_digest as _include_digest

_ROOT = os.path.dirname(os.path.abspath(__file__))
_JIT_CACHE_ENV = "ISOEXEC_DG_JIT_CACHE_DIR"
_JIT_SIGNATURE = f"isoexec-deep_gemm-{_UPSTREAM_REVISION[:12]}-include-{_include_digest()[:16]}"
if not os.environ.get(_JIT_CACHE_ENV):
    os.environ[_JIT_CACHE_ENV] = os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "isoexec", "deep_gemm_jit")
_C.init(_ROOT, _JIT_SIGNATURE)


def jit_identity() -> dict:
    """The include root, cache namespace and cache directories this copy's JIT compiles under."""
    return dict(_C.isoexec_jit_config())


def check_jit_identity() -> dict:
    """Fail closed unless the JIT compiles against this package's headers in its own namespace."""
    identity = jit_identity()
    expected = os.path.join(_ROOT, "include")
    problems = []
    if [os.path.normpath(p) for p in identity["include_dirs"]] != [os.path.normpath(expected)]:
        problems.append(f"include_dirs={identity['include_dirs']} (expected [{expected}])")
    if identity["env_prefix"] != "ISOEXEC_DG":
        problems.append(f"env_prefix={identity['env_prefix']}")
    if not identity["extra_signature"].startswith(_JIT_SIGNATURE + "+"):
        problems.append(f"extra_signature={identity['extra_signature']}")
    if problems:
        raise RuntimeError("Vendored DeepGEMM JIT identity is not this package's: " + "; ".join(problems))
    return identity


check_jit_identity()
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)

__version__ = '2.8.2'
