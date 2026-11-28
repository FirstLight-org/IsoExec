"""DeepSeek-V4.1 cache-record quantization (``deepseek_v41_cache_record_quant_v1``).

``reference.py`` is the CPU statement of the one quantizer of the V4.1 cache program (FP8 E4M3 codes with a
power-of-two UE8M0 scale per 32 or 128 channels; port specification: the vLLM 0.33 ``FLASHMLA_SPARSE_DSV41``
records). The V4.1 attention composition (``attention/deepseek_v41_cache_sm100``) implements it in CUDA for the
window-KV and compressed-KV records and the index keys. Importing this package never builds CUDA."""
