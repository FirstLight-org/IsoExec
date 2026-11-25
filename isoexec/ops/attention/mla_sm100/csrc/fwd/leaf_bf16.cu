// SPDX-License-Identifier: MIT
// IsoExec mla_sm100: the bf16 instantiation of the leaf kernel.
#include <cutlass/numeric_types.h>

#include "leaf.cuh"

namespace mla_sm100::fwd {
template void run_mla_sm100_leaf_kernel<cutlass::bfloat16_t>(FwdParams &params, const LeafTensors &tensors);
}
