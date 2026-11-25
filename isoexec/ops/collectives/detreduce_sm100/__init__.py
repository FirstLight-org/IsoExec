"""Deterministic, TP-size-invariant all-reduce of FP32 subtree roots over symmetric memory (sm100).

``ops.TreeReducer`` reduces each rank's FP32 subtree root by the fixed adjacent-pair tree in rank order
and rounds once to BF16: bitwise ``leaf_tree(stack(all_gather(root)))``. No build on import."""
