"""Expert-parallel data movement around vendored DeepGEMM MegaMoE (``_vendor/deep_gemm``). No build on import.

``mega_ep`` holds the symmetric output region of a MegaMoE workspace and the protocol of the kernels in
``csrc/mega_ep_io.cu``: one prep launch that fills MegaMoE's per-rank input buffer (rows, int64 top-k ids, FP32
weights, padding rows, the call counter) and one gather launch that reassembles replicated rows from every
rank's MegaMoE output over NVLink. Data movement only (the FP8 ``prep`` also applies the 1x32 UE8M0 activation
encoding of ``clamped_swiglu_sm100``). The expert programs that use it live with their recipes
(``moe/deepseek_v41_sm100/mega.py``: the BF16 actor recipe on MegaMoE).
"""
