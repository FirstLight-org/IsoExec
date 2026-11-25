"""GPU battery of the mla_sm100 forward: the mla_sm90 conformance relations, run on Blackwell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REL_ORDER = (
    "fwd_vs_reference",
    "logprob_contract",
    "batch_composition",
    "decode_vs_multirow",
    "chunked_prefill",
    "exec_override",
    "page_permutation",
    "row_placement",
    "repeat",
    "boundary_lengths",
    "nonfinite_unused_pages",
    "degenerate_shapes",
    "cuda_graph",
    "role_split_detected",
    # this package's own: the sm90 file's rel_cuda_graph cannot be driven from a Blackwell box
    "cuda_graph_sm100",
    # this package's own: explicit combine tree widths n = 1..20 (see _relations_sm100.py)
    "combine_tree_widths",
    # this package's own, round 4: the saturated uniform-decode cut rule (fwd.py), which no other
    # relation reaches because it needs >= 2 x workers row blocks of identical trees
    "saturated_decode_cut",
    # this package's own, round 5: the recipe-v2 lazy-rescale branch is exercised, and v2 equals v1
    # bitwise exactly where every tile's max advance exceeds the threshold
    "lazy_rescale_branches",
)


def _run_isolated(relation: str, args) -> dict:
    cmd = [sys.executable, "-c",
           "import json,sys;"
           "from isoexec.ops.attention.mla_sm90.tests._relations import Config, run_relation;"
           "from isoexec.ops.attention.mla_sm100.tests._relations_sm100 import RELATIONS_SM100,"
           " run_relation_sm100;"
           "from isoexec.ops.attention.mla_sm100.tests._backend import SM100Backend;"
           "rel, seg, heads, long_ctx, repeats, recipe = sys.argv[1], int(sys.argv[2]),"
           " int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]);"
           "b = SM100Backend(seg, recipe);"
           "cfg = Config(heads=heads, seg=seg, long_ctx=long_ctx, repeats=repeats);"
           "o = (run_relation_sm100 if rel in RELATIONS_SM100 else run_relation)(rel, b, cfg);"
           "print('@@' + json.dumps(o.as_dict()))",
           relation, str(args.seg), str(args.heads), str(args.long_ctx), str(args.repeats),
           str(args.recipe)]
    env = dict(os.environ)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    return {"relation": relation, "backend": "mla_sm100", "status": "error",
            "notes": [f"child exit {p.returncode}", p.stdout[-800:], p.stderr[-1500:]],
            "diffs": [], "skipped": [], "witness": {}, "metrics": {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg", type=int, default=None,
                    help="default: the schedule policy's span at spec.py's declarations (fwd.segment_span())")
    ap.add_argument("--heads", type=int, default=5)
    ap.add_argument("--long-ctx", type=int, default=8192 + 21)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--relations", default=",".join(_REL_ORDER))
    ap.add_argument("--report", default=None)
    ap.add_argument("--recipe", type=int, default=1, choices=(1, 2),
                    help="1: sm100_absorbed_segtree (default); 2: sm100_absorbed_segtree_v2")
    args = ap.parse_args(argv)
    if args.seg is None:
        from isoexec.ops.attention.mla_sm100.fwd import segment_span

        args.seg = segment_span()

    try:
        import torch

        from isoexec.core.arch import live_device_arch
    except Exception as exc:  # pragma: no cover
        print(f"INCOMPLETE: cannot import torch/isoexec: {exc}")
        return 2
    if not torch.cuda.is_available() or live_device_arch() != "sm100":
        print("INCOMPLETE: no SM100 device visible; the claim was not made here")
        return 2
    # This recipe carries no twin class (spec.TWIN_CLASS == ""), so there is no TWIN_PROOFS entry to
    # assert against: what this battery proves is the recipe's own determinism relations, not a
    # bitwise equality with another implementation.  If a twin is ever declared, this is where the
    # arch coverage must be asserted before the evidence is worth reading.
    from isoexec.ops.attention.mla_sm100 import spec as _spec

    if getattr(_spec, "TWIN_CLASS", ""):
        from isoexec.core import registry

        try:
            registry.assert_battery_covers_live_arch(_spec.TWIN_CLASS, "sm100")
        except Exception as exc:
            print(f"INCOMPLETE: {exc}")
            return 2

    rows, bad = [], []
    for rel in [r for r in args.relations.split(",") if r]:
        out = _run_isolated(rel, args)
        rows.append(out)
        status = out.get("status")
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP", "n/a": "N/A", "error": "ERROR"}.get(status, status)
        summary = out.get("notes") or out.get("skipped") or []
        ndiff = len(out.get("diffs", []))
        nbad = len([d for d in out.get("diffs", []) if not d.get("equal", True)])
        print(f"{mark:6} {rel:24} checks={ndiff:4} differ={nbad:4} {str(summary)[:200]}")
        if status in ("fail", "error"):
            bad.append(rel)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"args": vars(args), "relations": rows}, fh, indent=1)
    if bad:
        print(f"FAIL: {bad}")
        return 1
    print("PASS: every relation held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
