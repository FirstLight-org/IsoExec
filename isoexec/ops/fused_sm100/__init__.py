"""SM100 fused glue kernels for the native Flash models (row scatter, embedding expand, argmax, row scale).

Pure data movement and order-free selection: no reduction tree, no atomics. No kernel is built on import;
``ops.py`` builds the extension on first use.
"""
