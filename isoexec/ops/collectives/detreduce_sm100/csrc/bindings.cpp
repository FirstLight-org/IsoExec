// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include <vector>

namespace detreduce {
int64_t max_cta();
int64_t max_world();
int64_t tile_elems();
int64_t tag_grid();
int64_t tree_slot(int64_t cap, int64_t world);
void tree_allreduce(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& inbox_ptrs,
                    const std::vector<int64_t>& ybox_ptrs, int64_t cap, const std::vector<int64_t>& tbox_ptrs,
                    int64_t tcap, const std::vector<int64_t>& lbox_ptrs, int64_t lcap, const std::vector<int64_t>& lybox_ptrs,
                    int64_t lycap, const std::vector<int64_t>& flag_ptrs, at::Tensor& state, int64_t rank, int64_t two_shot,
                    int64_t grid);
void phased_entry(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& inbox_ptrs,
                    const std::vector<int64_t>& ybox_ptrs, int64_t cap, const std::vector<int64_t>& tbox_ptrs,
                    int64_t tcap, const std::vector<int64_t>& lbox_ptrs, int64_t lcap, const std::vector<int64_t>& lybox_ptrs,
                    int64_t lycap, const std::vector<int64_t>& flag_ptrs, at::Tensor& state, int64_t rank, int64_t two_shot,
                    int64_t grid);
std::vector<std::vector<int64_t>> phased_resources(int64_t world);
void fused_entry(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& lbox_ptrs, int64_t lcap,
                 const std::vector<int64_t>& lybox_ptrs, int64_t lycap, const std::vector<int64_t>& flag_ptrs, at::Tensor& state,
                 int64_t rank, int64_t two_shot, int64_t grid, int64_t unroll);
int64_t fused_two_shot_chunk(int64_t lcap, int64_t lycap, int64_t world, int64_t grid, int64_t unroll);
std::vector<int64_t> device_resources();
int64_t lamport2_chunk_host(int64_t lcap, int64_t lycap, int64_t world);
void tree_push(const at::Tensor& in, const std::vector<int64_t>& tbox_ptrs, int64_t tcap, const std::vector<int64_t>& lbox_ptrs,
               int64_t lcap, bool lamport, at::Tensor& state, int64_t rank);
void terms_combine(const at::Tensor& roots, const at::Tensor& dest, const at::Tensor& weights, at::Tensor& out,
                   const std::vector<int64_t>& terms_ptrs, int64_t terms_half, const std::vector<int64_t>& y_ptrs,
                   int64_t y_half, int64_t cap_rows, int64_t cap_y, const std::vector<int64_t>& tterms_ptrs,
                   int64_t tterms_half, int64_t cap_t, const std::vector<int64_t>& flag_ptrs,
                   at::Tensor& state, int64_t rank, int64_t two_shot, int64_t grid);
void moe_prep(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& lidx, int64_t shared_group, int64_t s0,
              int64_t s1, at::Tensor& codes, at::Tensor& sa, at::Tensor& offsets, at::Tensor& dest,
              const c10::optional<at::Tensor>& done);
}  // namespace detreduce

namespace detreduce {
void leaf_fold_bf16(const at::Tensor &planes, at::Tensor &out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_cta", &detreduce::max_cta);
  m.def("leaf_fold_bf16", &detreduce::leaf_fold_bf16);
  m.def("max_world", &detreduce::max_world);
  m.def("tile_elems", &detreduce::tile_elems);
  m.def("tag_grid", &detreduce::tag_grid);
  m.def("tree_slot", &detreduce::tree_slot);
  m.def("lamport2_chunk", &detreduce::lamport2_chunk_host);
  m.def("phased_entry", &detreduce::phased_entry);
  m.def("phased_resources", &detreduce::phased_resources);
  m.def("fused_entry", &detreduce::fused_entry);
  m.def("fused_two_shot_chunk", &detreduce::fused_two_shot_chunk);
  m.def("device_resources", &detreduce::device_resources);
  m.def("tree_allreduce", &detreduce::tree_allreduce);
  m.def("tree_push", &detreduce::tree_push);
  m.def("terms_combine", &detreduce::terms_combine);
  m.def("moe_prep", &detreduce::moe_prep);
}
