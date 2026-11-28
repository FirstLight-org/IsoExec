// Copied from attention/deepseek_v41_cache_sm100/csrc/pdl.cuh (same program; this extension owns its copy).
// Programmatic dependent launch for this extension's small kernels (launch structure only).
//
// Every kernel launched through pdl::launch begins with pdl::begin(): griddepcontrol.wait (no global
// memory is read or written before the previous kernel in the stream has completed and its memory is
// visible) and then griddepcontrol.launch_dependents (the NEXT kernel's CTAs may be scheduled from
// here on; a PDL-launched successor waits in its own griddepcontrol.wait for this grid's completion).
// So every kernel still runs strictly after its predecessor and before its successor; only launch
// latency and CTA scheduling overlap. No arithmetic, no memory order, no output bit changes.
// Without the launch attribute (ISOEXEC_SMALL_PDL=0, or a predecessor that is not a kernel) both
// instructions are no-ops. Trigger-after-wait bounds the overlap to one kernel ahead.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdlib>
#include <utility>

namespace pdl {

__device__ __forceinline__ void begin() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

inline bool enabled() {           // read per launch (cheap; captured launches keep the value they were made with)
    const char *e = std::getenv("ISOEXEC_SMALL_PDL");
    return !(e && e[0] == '0');
}

template <typename... KArgs, typename... Args>
inline void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                   Args &&...args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = enabled() ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace pdl
