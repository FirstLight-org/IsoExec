// The compiled configuration table. segments.py::_CONFIGS mirrors it and ops.config_ids() checks both
// directions, so the two cannot drift. Grouped by translation unit only to parallelise the build.
//
// X(name, BM, BN, CM, CN, schedule, ElementD, out_fp32, levels, swap, stages)
//   swap: tokens on the wgmma N side (see Gemm::kSwap); stages: 0 = all of shared memory
//   schedule: WS (one consumer warpgroup, one tile per CTA), PP (pingpong, persistent),
//             CO (cooperative, persistent)
#pragma once

// bf16 out, single chain per batch: the plain mm and the pik leaf wire
#define ISOEXEC_MM_SM90_TU0(X)                                   \
  X(ws_64x16, 64, 16, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 0, 0)       \
  X(ws_64x32, 64, 32, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 0, 0)       \
  X(ws_64x64, 64, 64, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 0, 0)       \
  X(ws_64x128, 64, 128, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 0, 0)

#define ISOEXEC_MM_SM90_TU1(X)                                   \
  X(pp_64x128, 64, 128, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 0, 0)     \
  X(pp_64x256, 64, 256, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 0, 0)     \
  X(pp_128x128, 128, 128, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 0, 0)   \
  X(co_128x64, 128, 64, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)

#define ISOEXEC_MM_SM90_TU2(X)                                   \
  X(co_128x128, 128, 128, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)   \
  X(co_128x256, 128, 256, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)   \
  X(co_128x256_c2, 128, 256, 2, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)

#define ISOEXEC_MM_SM90_TU3(X)                                   \
  X(co_256x128_c2, 256, 128, 1, 2, CO, cutlass::bfloat16_t, 0, 0, 0, 0) \
  X(co_128x128_c2, 128, 128, 2, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0) \
  X(pp_128x128_c2, 128, 128, 2, 1, PP, cutlass::bfloat16_t, 0, 0, 0, 0)

// fp32 out, single chain per batch: segment roots for the split execution (combine kernel after)
#define ISOEXEC_MM_SM90_TU4(X)                                   \
  X(ws_64x64_f32, 64, 64, 1, 1, WS, float, 1, 0, 0, 0)                 \
  X(co_128x128_f32, 128, 128, 1, 1, CO, float, 1, 0, 0, 0)             \
  X(co_128x256_f32, 128, 256, 2, 1, CO, float, 1, 0, 0, 0)

// onchip fold, up to 8 segments per CTA: bf16 root (the mm at S > 1) and fp32 node (pik node wire)
#define ISOEXEC_MM_SM90_TU5(X)                                   \
  X(ws_64x32_t3, 64, 32, 1, 1, WS, cutlass::bfloat16_t, 0, 3, 0, 0)    \
  X(ws_64x64_t3, 64, 64, 1, 1, WS, cutlass::bfloat16_t, 0, 3, 0, 0)    \
  X(co_128x64_t3, 128, 64, 1, 1, CO, cutlass::bfloat16_t, 0, 3, 0, 0)

#define ISOEXEC_MM_SM90_TU6(X)                                   \
  X(ws_64x32_t3_f32, 64, 32, 1, 1, WS, float, 1, 3, 0, 0)              \
  X(ws_64x64_t3_f32, 64, 64, 1, 1, WS, float, 1, 3, 0, 0)              \
  X(ws_64x128_t3_f32, 64, 128, 1, 1, WS, float, 1, 3, 0, 0)            \
  X(co_128x64_t3_f32, 128, 64, 1, 1, CO, float, 1, 3, 0, 0)

// swapped operands (tokens on the wgmma N side) for decode M: sw_<weight rows>x<tokens>_s<stages>
#define ISOEXEC_MM_SM90_TU7(X)                                                \
  X(sw_64x8_s6, 64, 8, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 6)               \
  X(sw_64x16_s6, 64, 16, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 6)             \
  X(sw_64x32_s6, 64, 32, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 6)             \
  X(sw_64x64_s6, 64, 64, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 6)

#define ISOEXEC_MM_SM90_TU8(X)                                                \
  X(sw_64x16_s10, 64, 16, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 10)           \
  X(sw_64x32_s10, 64, 32, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 10)           \
  X(swpp_64x16, 64, 16, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 1, 0)              \
  X(swpp_64x32, 64, 32, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 1, 0)              \
  X(swpp_64x64, 64, 64, 1, 1, PP, cutlass::bfloat16_t, 0, 0, 1, 0)

#define ISOEXEC_MM_SM90_TU9(X)                                                \
  X(swco_128x16, 128, 16, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 1, 0)            \
  X(swco_128x32, 128, 32, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 1, 0)            \
  X(swco_128x64, 128, 64, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 1, 0)            \
  X(sw_64x128_s6, 64, 128, 1, 1, WS, cutlass::bfloat16_t, 0, 0, 1, 6)

// swapped onchip fold, fp32 node (pik node wire at decode M)
#define ISOEXEC_MM_SM90_TU10(X)                                               \
  X(sw_64x16_s6_t3_f32, 64, 16, 1, 1, WS, float, 1, 3, 1, 6)                    \
  X(sw_64x32_s6_t3_f32, 64, 32, 1, 1, WS, float, 1, 3, 1, 6)                    \
  X(swpp_64x32_t3_f32, 64, 32, 1, 1, PP, float, 1, 3, 1, 0)                     \
  X(swpp_64x64_t3_f32, 64, 64, 1, 1, PP, float, 1, 3, 1, 0)                     \
  X(swco_128x32_t3_f32, 128, 32, 1, 1, CO, float, 1, 3, 1, 0)

// wave-quantization tiles for mid M
#define ISOEXEC_MM_SM90_TU11(X)                                               \
  X(co_256x128, 256, 128, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)          \
  X(co_128x192, 128, 192, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)          \
  X(co_256x64, 256, 64, 1, 1, CO, cutlass::bfloat16_t, 0, 0, 0, 0)

// depth-1 fold (two segments per CTA: pik's m = 2 at TP4, G = 8), fp32 node
#define ISOEXEC_MM_SM90_TU12(X)                                               \
  X(sw_64x16_s10_t1_f32, 64, 16, 1, 1, WS, float, 1, 1, 1, 10)                \
  X(swpp_64x32_t1_f32, 64, 32, 1, 1, PP, float, 1, 1, 1, 0)                   \
  X(ws_64x128_t1_f32, 64, 128, 1, 1, WS, float, 1, 1, 0, 0)                   \
  X(pp_64x128_t1_f32, 64, 128, 1, 1, PP, float, 1, 1, 0, 0)                   \
  X(co_128x128_t1_f32, 128, 128, 1, 1, CO, float, 1, 1, 0, 0)

#define ISOEXEC_MM_SM90_NTU 13

#define ISOEXEC_MM_SM90_ALL(X)                                   \
  ISOEXEC_MM_SM90_TU0(X) ISOEXEC_MM_SM90_TU1(X) ISOEXEC_MM_SM90_TU2(X) ISOEXEC_MM_SM90_TU3(X) \
  ISOEXEC_MM_SM90_TU4(X) ISOEXEC_MM_SM90_TU5(X) ISOEXEC_MM_SM90_TU6(X) ISOEXEC_MM_SM90_TU7(X) \
  ISOEXEC_MM_SM90_TU8(X) ISOEXEC_MM_SM90_TU9(X) ISOEXEC_MM_SM90_TU10(X) ISOEXEC_MM_SM90_TU11(X) \
  ISOEXEC_MM_SM90_TU12(X)

#define ISOEXEC_MM_SM90_KS_WS cutlass::gemm::KernelTmaWarpSpecialized
#define ISOEXEC_MM_SM90_ES_WS cutlass::epilogue::TmaWarpSpecialized
#define ISOEXEC_MM_SM90_KS_PP cutlass::gemm::KernelTmaWarpSpecializedPingpong
#define ISOEXEC_MM_SM90_ES_PP cutlass::epilogue::TmaWarpSpecialized
#define ISOEXEC_MM_SM90_KS_CO cutlass::gemm::KernelTmaWarpSpecializedCooperative
#define ISOEXEC_MM_SM90_ES_CO cutlass::epilogue::TmaWarpSpecializedCooperative
#define ISOEXEC_MM_SM90_SN_WS "ws"
#define ISOEXEC_MM_SM90_SN_PP "pingpong"
#define ISOEXEC_MM_SM90_SN_CO "cooperative"

#define ISOEXEC_MM_SM90_TYPE(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST)                                     \
  ::isoexec_mm_sm90::Gemm<BM, BN, CM, CN, ISOEXEC_MM_SM90_KS_##S, ISOEXEC_MM_SM90_ES_##S, ED, LV, SW, ST>

#define ISOEXEC_MM_SM90_DEFINE(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST)                                   \
  namespace isoexec_mm_sm90 {                                                                          \
  int run_##name(LaunchArgs const& a) { return ISOEXEC_MM_SM90_TYPE(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST)::run(a); } \
  int stages_##name() { return ISOEXEC_MM_SM90_TYPE(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST)::Stages; } \
  }

#define ISOEXEC_MM_SM90_DECLARE(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST)                                  \
  namespace isoexec_mm_sm90 {                                                                          \
  int run_##name(LaunchArgs const& a);                                                                 \
  int stages_##name();                                                                                 \
  }
