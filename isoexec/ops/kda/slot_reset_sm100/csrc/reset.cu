// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAException.h>

template <typename Slot, typename Pos>
__global__ void reset_slots(uint16_t* conv, uint8_t* rec, const Slot* slots,
                           const Pos* positions, int64_t rows, int64_t capacity,
                           int64_t null_slot, int64_t D, int64_t C, int64_t H,
                           int64_t B, int64_t cs0, int64_t cs1, int64_t cs2,
                           int64_t rs0, int64_t rs1, int64_t rs2) {
    const int64_t row = blockIdx.x;
    const int64_t slot = slots[row];
    if (slot < 0 || slot == null_slot) return;
    CUDA_KERNEL_ASSERT(slot < capacity);
    // One block owns a request's state. Validate every live owner even on
    // continuation rows; duplicate metadata must fail rather than race stores.
    for (int64_t other = threadIdx.x; other < rows; other += blockDim.x) {
        CUDA_KERNEL_ASSERT(other == row || slots[other] != slot);
    }
    __syncthreads();
    if (positions[row] != 0) return;
    for (int64_t i = threadIdx.x; i < D * C; i += blockDim.x)
        conv[slot * cs0 + (i / C) * cs1 + (i % C) * cs2] = 0;
    for (int64_t i = threadIdx.x; i < H * B; i += blockDim.x)
        rec[slot * rs0 + (i / B) * rs1 + (i % B) * rs2] = 0;
}

template <typename Slot, typename Pos>
void launch(torch::Tensor conv, torch::Tensor rec, torch::Tensor slots,
            torch::Tensor positions, int64_t null_slot) {
    reset_slots<Slot, Pos><<<slots.numel(), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint16_t*>(conv.data_ptr()), rec.data_ptr<uint8_t>(),
        slots.data_ptr<Slot>(), positions.data_ptr<Pos>(), slots.numel(), conv.size(0),
        null_slot, conv.size(1), conv.size(2), rec.size(1), rec.size(2),
        conv.stride(0), conv.stride(1), conv.stride(2), rec.stride(0), rec.stride(1), rec.stride(2));
}

void reset(torch::Tensor conv, torch::Tensor rec, torch::Tensor slots,
           torch::Tensor positions, int64_t null_slot) {
    TORCH_CHECK(conv.is_cuda() && rec.device() == conv.device() &&
                slots.device() == conv.device() && positions.device() == conv.device(),
                "KDA slot reset requires one CUDA device");
    TORCH_CHECK(conv.dim() == 3 && rec.dim() == 3 && conv.size(0) == rec.size(0),
                "KDA slot reset requires slot-major 3D states with equal capacities");
    TORCH_CHECK(conv.scalar_type() == at::kBFloat16 && rec.scalar_type() == at::kByte,
                "KDA slot reset requires BF16 convolution and byte records");
    TORCH_CHECK(slots.dim() == 1 && positions.dim() == 1 && slots.numel() == positions.numel() &&
                slots.is_contiguous() && positions.is_contiguous(),
                "KDA slot reset requires contiguous equal-length slot/position vectors");
    TORCH_CHECK((slots.scalar_type() == at::kInt || slots.scalar_type() == at::kLong) &&
                (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
                "KDA slot reset requires int32/int64 slots and positions");
    if (!slots.numel()) return;
    c10::cuda::CUDAGuard guard(conv.device());
    C10_CUDA_CHECK(c10::cuda::SetDevice(conv.get_device(), true));
    if (slots.scalar_type() == at::kInt) {
        if (positions.scalar_type() == at::kInt) launch<int32_t, int32_t>(conv, rec, slots, positions, null_slot);
        else launch<int32_t, int64_t>(conv, rec, slots, positions, null_slot);
    } else {
        if (positions.scalar_type() == at::kInt) launch<int64_t, int32_t>(conv, rec, slots, positions, null_slot);
        else launch<int64_t, int64_t>(conv, rec, slots, positions, null_slot);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Every KDA layer of one step in one launch: block (row, layer) performs reset_slots' stores for that layer's states.
// All layers share the step's slots/positions and one state geometry (shapes and strides); only the base
// addresses differ. Ownership validation runs once per row (layer 0's block), as each single-layer launch would.
constexpr int kMaxLayers = 64;
struct LayerStates {
    uint16_t* conv[kMaxLayers];
    uint8_t* rec[kMaxLayers];
};

template <typename Slot, typename Pos>
__global__ void reset_slots_layers(const __grid_constant__ LayerStates st, const Slot* slots,
                                   const Pos* positions, int64_t rows, int64_t capacity,
                                   int64_t null_slot, int64_t D, int64_t C, int64_t H,
                                   int64_t B, int64_t cs0, int64_t cs1, int64_t cs2,
                                   int64_t rs0, int64_t rs1, int64_t rs2) {
    const int64_t row = blockIdx.x;
    const int layer = blockIdx.y;
    const int64_t slot = slots[row];
    if (slot < 0 || slot == null_slot) return;
    CUDA_KERNEL_ASSERT(slot < capacity);
    if (layer == 0) {
        for (int64_t other = threadIdx.x; other < rows; other += blockDim.x) {
            CUDA_KERNEL_ASSERT(other == row || slots[other] != slot);
        }
    }
    if (positions[row] != 0) return;
    uint16_t* conv = st.conv[layer];
    uint8_t* rec = st.rec[layer];
    for (int64_t i = threadIdx.x; i < D * C; i += blockDim.x)
        conv[slot * cs0 + (i / C) * cs1 + (i % C) * cs2] = 0;
    for (int64_t i = threadIdx.x; i < H * B; i += blockDim.x)
        rec[slot * rs0 + (i / B) * rs1 + (i % B) * rs2] = 0;
}

template <typename Slot, typename Pos>
void launch_layers(const std::vector<torch::Tensor>& convs, const std::vector<torch::Tensor>& recs,
                   torch::Tensor slots, torch::Tensor positions, int64_t null_slot) {
    LayerStates st{};
    for (size_t l = 0; l < convs.size(); ++l) {
        st.conv[l] = reinterpret_cast<uint16_t*>(convs[l].data_ptr());
        st.rec[l] = recs[l].data_ptr<uint8_t>();
    }
    const auto& conv = convs[0];
    const auto& rec = recs[0];
    reset_slots_layers<Slot, Pos><<<dim3(slots.numel(), convs.size()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        st, slots.data_ptr<Slot>(), positions.data_ptr<Pos>(), slots.numel(), conv.size(0),
        null_slot, conv.size(1), conv.size(2), rec.size(1), rec.size(2),
        conv.stride(0), conv.stride(1), conv.stride(2), rec.stride(0), rec.stride(1), rec.stride(2));
}

void reset_layers(std::vector<torch::Tensor> convs, std::vector<torch::Tensor> recs, torch::Tensor slots,
                  torch::Tensor positions, int64_t null_slot) {
    TORCH_CHECK(!convs.empty() && convs.size() == recs.size() && convs.size() <= kMaxLayers,
                "KDA layer slot reset takes 1..64 (conv, record) state pairs");
    for (size_t l = 0; l < convs.size(); ++l) {
        const auto& c = convs[l];
        const auto& r = recs[l];
        TORCH_CHECK(c.is_cuda() && r.device() == convs[0].device() && c.device() == convs[0].device(),
                    "KDA layer slot reset requires one CUDA device");
        TORCH_CHECK(c.dim() == 3 && r.dim() == 3 && c.size(0) == r.size(0), "slot-major 3D states with equal capacities");
        TORCH_CHECK(c.scalar_type() == at::kBFloat16 && r.scalar_type() == at::kByte, "BF16 convolution and byte records");
        TORCH_CHECK(c.sizes() == convs[0].sizes() && c.strides() == convs[0].strides() &&
                    r.sizes() == recs[0].sizes() && r.strides() == recs[0].strides(),
                    "KDA layer slot reset requires one state geometry across layers");
    }
    TORCH_CHECK(slots.device() == convs[0].device() && positions.device() == convs[0].device() &&
                slots.dim() == 1 && positions.dim() == 1 && slots.numel() == positions.numel() &&
                slots.is_contiguous() && positions.is_contiguous(),
                "KDA slot reset requires contiguous equal-length slot/position vectors");
    TORCH_CHECK((slots.scalar_type() == at::kInt || slots.scalar_type() == at::kLong) &&
                (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
                "KDA slot reset requires int32/int64 slots and positions");
    if (!slots.numel()) return;
    c10::cuda::CUDAGuard guard(convs[0].device());
    C10_CUDA_CHECK(c10::cuda::SetDevice(convs[0].get_device(), true));
    if (slots.scalar_type() == at::kInt) {
        if (positions.scalar_type() == at::kInt) launch_layers<int32_t, int32_t>(convs, recs, slots, positions, null_slot);
        else launch_layers<int32_t, int64_t>(convs, recs, slots, positions, null_slot);
    } else {
        if (positions.scalar_type() == at::kInt) launch_layers<int64_t, int32_t>(convs, recs, slots, positions, null_slot);
        else launch_layers<int64_t, int64_t>(convs, recs, slots, positions, null_slot);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reset", &reset);
    m.def("reset_layers", &reset_layers);
}
