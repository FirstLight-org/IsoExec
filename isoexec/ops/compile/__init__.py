"""Pinned compiled regions for backward-only spans, served from a ledger of offline verdicts.

Deliberately empty of side effects, like every other ``ops/`` package: importing it must not
compile, register or rebind anything.

``region_gate`` compiles a region under the pinned configuration, ``cfgpin`` pins the Inductor
configuration, ``decomp`` holds the bitwise decomposition table, and ``bwd_compile`` is the
backward-only ``call_region`` entry point that ``ops/gdn`` calls. It lives under ``ops/`` because an admitted
compiled span is an impl like any other (00 section 2.9, audit 2026-09-05 section 5.1): it passes
the same bitwise gate a hand-written fused kernel passes, and it is bound by the same op table.
"""
