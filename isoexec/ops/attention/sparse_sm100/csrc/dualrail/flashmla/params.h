// Vendored from FlashMLA 2e5429f csrc/params.h (MIT, see ../LICENSE-FlashMLA.txt), CUDA-only and STRIPPED to the
// types the dual-rail leaf uses. The split-KV scheduler metadata struct, the decode / prefill parameter
// structs and every Ascend path are removed: the IsoExec leaf has no batch-dependent scheduling.
#pragma once

#include <cstdint>
#include <type_traits>
#include <cutlass/bfloat16.h>
#include <cute/container/alignment.hpp>

using bf16 = cutlass::bfloat16_t;
using stream_t = cudaStream_t;
#define FLASH_MLA_GM

enum class ModelType {
    V41         // DeepSeek V4.1 fp8 token: 512 e4m3 + 16 ue8m0 (block 32), 528 B (IsoExec: upstream's V41_FP4 removed)
};
