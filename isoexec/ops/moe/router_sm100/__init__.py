"""SM100 DeepSeek-class MoE routers: grouped sigmoid + bias (V3-class) and sqrt-softplus + bias
(V4-class), each as a select-only kernel over caller logits and as one kernel fused with the router
GEMM. No build on import."""
