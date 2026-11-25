"""Run MLA conformance against selected backends and write a report with --report.
Import failures are skips; any mla_sm90 failure or error sets exit status 1."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from isoexec.ops.attention.mla_sm90.tests._harness import load_backend
from isoexec.ops.attention.mla_sm90.tests._relations import RELATIONS, SENSITIVITY_RELATIONS, Config, Outcome, run_relation

DEFAULT_BACKENDS = "mla_sm90,flashmla_head,flashmla_vllm,fa3_ns0,fa3_ns1"


def _run_isolated(bname: str, relation: str, args) -> Outcome:
    """Run one relation in a child interpreter; a hang/crash becomes an ERROR outcome."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    cmd = [sys.executable, "-m", "isoexec.ops.attention.mla_sm90.tests.conformance", "--backends", bname, "--relations",
           relation, "--heads", str(args.heads), "--seg", str(args.seg), "--ctx", str(args.ctx), "--repeats",
           str(args.repeats), "--report", path, "--_child", "1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        with open(path) as f:
            d = json.load(f)["results"][bname][relation]
        o = Outcome(relation, bname, status=d["status"], skipped=d["skipped"], witness=d["witness"],
                    metrics=d["metrics"], notes=d["notes"], seconds=d["seconds"])
        from isoexec.ops.attention.mla_sm90.tests._harness import Diff

        o.diffs = [Diff(**x) for x in d["diffs"]]
        return o
    except subprocess.TimeoutExpired:
        return Outcome(relation, bname, status="error", notes=[f"timeout after {args.timeout}s (hang)"])
    except Exception as exc:
        tail = proc.stderr[-800:] if "proc" in locals() else ""
        return Outcome(relation, bname, status="error",
                       notes=[f"child failed (rc={getattr(locals().get('proc'), 'returncode', None)}): {exc!r}", tail])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default=DEFAULT_BACKENDS)
    ap.add_argument("--relations", default="all", help="comma list, 'all' or 'sensitivity'")
    ap.add_argument("--heads", type=int, default=5)
    ap.add_argument("--seg", type=int, default=int(os.environ.get("ISOEXEC_MLA_SM90_SEG", 512)))
    ap.add_argument("--ctx", type=int, default=8192 + 21, help="long-context length for composition relations")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--report", default=None, help="write JSON results here")
    ap.add_argument("--verbose", action="store_true", help="print every failing check")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run relations in this process (default: one child process per relation, so a hung or "
                         "crashing kernel is reported as ERROR instead of freezing the suite)")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per relation when isolated")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA unavailable: nothing to run")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    rels = (list(RELATIONS) if args.relations == "all" else
            list(SENSITIVITY_RELATIONS) if args.relations == "sensitivity" else args.relations.split(","))
    cfg = Config(heads=args.heads, seg=args.seg, long_ctx=args.ctx, repeats=args.repeats)
    env = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
           "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "config": vars(cfg) if hasattr(cfg, "__dict__") else str(cfg)}
    if not args._child:
        print(json.dumps(env), flush=True)

    results: dict[str, dict[str, Outcome]] = {}
    load_errors = {}
    for bname in args.backends.split(","):
        backend, why = load_backend(bname, args.seg)
        results[bname] = {}
        if backend is None:
            load_errors[bname] = why
            for r in rels:
                results[bname][r] = Outcome(r, bname, status="skip", notes=[f"backend unavailable: {why}"])
            print(f"[{bname}] SKIP: {why}", flush=True)
            continue
        for r in rels:
            o = (run_relation(r, backend, cfg) if args.no_isolate or args._child
                 else _run_isolated(bname, r, args))
            results[bname][r] = o
            if args._child:
                continue
            print(f"[{bname}] {r:26s} {o.status.upper():6s} {o.seconds:7.1f}s  {o.summary()}", flush=True)
            if args.verbose and o.status in ("fail", "error"):
                for d in o.failures:
                    print(f"      - {d.name}: {d.detail}")
                for n in o.notes:
                    print(f"      * {n}")
            torch.cuda.empty_cache()

    if args._child:
        with open(args.report, "w") as f:
            json.dump({"results": {n: {r: o.as_dict() for r, o in v.items()} for n, v in results.items()}}, f,
                      default=str)
        return 0
    identities = {n: sorted({json.dumps(o.witness.get("kernel_identity"), sort_keys=True)
                             for o in v.values() if o.witness.get("kernel_identity")}) for n, v in results.items()}
    names = list(results)
    width = max(12, *(len(n) for n in names))
    print("\n" + "relation".ljust(26) + "".join(n.ljust(width + 2) for n in names))
    for r in rels:
        print(r.ljust(26) + "".join(results[n][r].status.upper().ljust(width + 2) for n in names))
    for n in names:
        for r in rels:
            o = results[n][r]
            if o.status in ("fail", "error"):
                print(f"\n{n} / {r}: {o.summary()}")
                for n2 in o.notes[:3]:
                    print(f"    {n2.splitlines()[0] if n2 else ''}")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"env": env, "load_errors": load_errors, "kernel_identities": identities,
                       "results": {n: {r: o.as_dict() for r, o in v.items()} for n, v in results.items()}},
                      f, indent=1, default=str)
        print(f"\nwrote {args.report}")
    mixed = False
    for n, ids in identities.items():
        if len(ids) > 1:
            mixed = True
            print(f"\nMIXED KERNEL IDENTITIES for {n}: {len(ids)} digests across outcomes; this run is NOT evidence:")
            for i in ids:
                print(f"    {i}")
        elif ids:
            print(f"\n{n} kernel identity: {ids[0]}")
    bad = [r for r, o in results.get("mla_sm90", {}).items() if o.status in ("fail", "error")]
    return 1 if bad or mixed else 0


if __name__ == "__main__":
    sys.exit(main())
