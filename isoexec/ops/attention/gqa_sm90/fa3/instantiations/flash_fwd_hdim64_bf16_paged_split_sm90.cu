// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// Upstream's generate_kernels.py emits this instantiation; IsoExec builds it for the segment split (block.h).

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, true, true, false, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
