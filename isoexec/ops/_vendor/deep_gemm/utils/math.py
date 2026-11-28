def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# IsoExec (pruned to BF16): the FP8/FP4 cast, pack and dequantize helpers are not part of this copy
