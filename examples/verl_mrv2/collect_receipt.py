"""Assemble one case receipt from an attempt's compact receipts and logs (no full rows are read or kept).

PASS requires: terminal exit 0; the trainer's ``complete`` lifecycle event; W0 and W1 coverage gates;
exactly one optimizer update on every trainer rank with changed parameters; every inference worker
acknowledging W0 and W1 on vLLM Model Runner V2; V2 full-row producer serves; V2 CUDA graph captures
and replays; and, per case, the axis evidence the plan names. Missing evidence is reported per item
as ``missing``, never as passed. Negative controls are recorded as ``not_run`` by policy.

Usage: python examples/verl_mrv2/collect_receipt.py --run-dir RUN_DIR --case N --out receipt.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOG_PATTERNS = {
    "v2_logprob_hook": re.compile(r"\[ISOEXEC-SAMPLER-LOGPROBS-V2\] pid=(\d+) .*?calls=(\d+) full_rows=(\d+) rowinv_owned=(\d+) incumbent=(\d+)"),
    "v2_cudagraph": re.compile(r"\[ISOEXEC-V2-CUDAGRAPH\] pid=(\d+) captures=(\d+) replays=(\d+)"),
    "ep_engine": re.compile(r"\[ISOEXEC-MOE-EP-ENGINE\] pid=(\d+) ep_rank=(\d+) ep=(\d+) .*?remote_routes=(\d+)"),
    "cp_attention": re.compile(r"\[ISOEXEC-CP-ATTN\] pid=(\d+) pcp_rank=(\d+) dcp_rank=(\d+) stats=(\{[^}]*\})"),
    "v2_requests": re.compile(r"\[ISOEXEC-V2-REQUESTS\] pid=(\d+) dp_rank=(\d+) new_requests=(\d+)"),
    "wandb_url": re.compile(r"(https://wandb\.ai/[\w\-.]+/[\w\-.]+/runs/[\w]+)"),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _last_by_pid(matches: list[tuple]) -> dict:
    last = {}
    for match in matches:
        last[match[0]] = match
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipts = args.run_dir / "evidence" / "receipts"
    log_text = "\n".join(path.read_text(errors="replace") for path in sorted((args.run_dir / "logs").glob("*.log")))
    lifecycle = {event["event"]: event for event in _jsonl(receipts / "lifecycle.jsonl")}
    terminal = json.loads((receipts / "terminal.json").read_text()) if (receipts / "terminal.json").exists() else {}
    found = {name: pattern.findall(log_text) for name, pattern in LOG_PATTERNS.items()}
    items: dict[str, dict] = {}

    def item(name: str, ok: bool | None, detail) -> None:
        items[name] = {"status": "missing" if ok is None else ("pass" if ok else "fail"), "detail": detail}

    item("terminal_exit_zero", terminal.get("exit_code") == 0 if terminal else None, terminal)
    item("lifecycle_complete", lifecycle.get("complete", {}).get("passed") is True if lifecycle else None,
         lifecycle.get("complete"))
    for version in (0, 1):
        sync, compare = lifecycle.get(f"w{version}_sync"), lifecycle.get(f"w{version}_compare")
        item(f"w{version}_worker_acks", None if sync is None else sync["workers"] > 0,
             None if sync is None else {"workers": sync["workers"], "runners": sync["runner_classes"]})
        item(f"w{version}_full_row_compare", None if compare is None else compare["rows"] > 0, compare)
    # Train/inference mismatch, from every rank's full-vocabulary compare records (not configuration).
    compares = [event for path in sorted(receipts.glob("compare-rank*.jsonl")) for event in _jsonl(path)
                if event.get("event") == "compare"]
    mismatch = {}
    for version in (0, 1):
        events = [event for event in compares if event["version"] == version]
        mismatch[f"w{version}"] = {
            "row_slices_compared": sum(event["rows"] for event in events),
            "engine_vocab": sorted({event["engine_vocab"] for event in events}),
            "mismatched_rows": sum(event.get("mismatched_rows", 0) or 0 for event in events),
            "max_abs_diff": max((event.get("max_abs_diff", 0.0) or 0.0 for event in events), default=None),
            "first_mismatch": next(({k: event[k] for k in ("key", "first_bad_row", "first_token", "trainer_u32",
                                                          "engine_u32", "mismatch_count")}
                                    for event in events if event["first_bad_row"] is not None), None),
        }
    full = all(mismatch[f"w{v}"]["row_slices_compared"] > 0 for v in (0, 1))
    item("train_inference_mismatch", None if not compares else (full and all(
        mismatch[f"w{v}"]["mismatched_rows"] == 0 and mismatch[f"w{v}"]["first_mismatch"] is None for v in (0, 1))),
        mismatch)
    signal = lifecycle.get("learning_signal")
    item("learning_signal", None if signal is None else signal["advantage_abs_max"] > 0, signal)
    optimizer = lifecycle.get("optimizer")
    item("one_optimizer_update", None if optimizer is None else True, optimizer)
    runners = sorted({runner for v in (0, 1) for runner in (lifecycle.get(f"w{v}_sync") or {}).get("runner_classes", [])})
    item("model_runner_v2", None if not runners else all(r.startswith("vllm.v1.worker.gpu.model_runner.") for r in runners),
         runners)
    # Every inference worker that acknowledged W1 must show its own census lines; a missing PID is
    # "missing", never inferred from another worker.
    acks_dir = args.run_dir / "evidence" / "acks" / "v1"
    acks = [json.loads(path.read_text()) for path in sorted(acks_dir.glob("*.json"))] if acks_dir.exists() else []
    expected_workers = 1 if args.case == 1 else 4
    worker_pids = {str(ack["pid"]) for ack in acks}
    last_stage = max((ack["pp_rank"] for ack in acks), default=0)
    sampler_pids = {str(ack["pid"]) for ack in acks if ack["pp_rank"] == last_stage}
    complete_acks = (len(acks) == expected_workers and len(worker_pids) == expected_workers
                     and all(ack["version"] == 1 for ack in acks))
    hooks = _last_by_pid(found["v2_logprob_hook"])
    graphs = _last_by_pid(found["v2_cudagraph"])
    missing_hooks = sorted(sampler_pids - set(hooks))
    missing_graphs = sorted(worker_pids - set(graphs))
    hook_ok = None if not complete_acks or missing_hooks or not sampler_pids else all(
        int(hooks[pid][3]) > 0 and int(hooks[pid][4]) == 0 for pid in sampler_pids)
    graph_ok = None if not complete_acks or missing_graphs else all(
        int(graphs[pid][1]) > 0 and int(graphs[pid][2]) > 0 for pid in worker_pids)
    item("v2_full_row_producer", hook_ok, {
        "expected_pids": sorted(sampler_pids), "missing_pids": missing_hooks,
        "observed": {pid: {"full_rows": int(m[2]), "rowinv_owned": int(m[3]), "incumbent": int(m[4])}
                     for pid, m in hooks.items()}})
    item("cuda_graph_capture_replay", graph_ok, {
        "expected_pids": sorted(worker_pids), "missing_pids": missing_graphs,
        "observed": {pid: {"captures": int(m[1]), "replays": int(m[2])} for pid, m in graphs.items()}})
    if args.case == 2:
        ep = _last_by_pid(found["ep_engine"])
        requests = _last_by_pid(found["v2_requests"])
        by_dp: dict[int, int] = {}
        for match in requests.values():
            by_dp[int(match[1])] = max(by_dp.get(int(match[1]), 0), int(match[2]))
        item("dp_request_distribution", None if not requests else len(by_dp) >= 2 and all(v > 0 for v in by_dp.values()),
             by_dp)
        item("ep_remote_dispatch", None if not ep else all(int(m[3]) > 0 for m in ep.values()),
             {pid: {"ep_rank": int(m[1]), "remote_routes": int(m[3])} for pid, m in ep.items()})
    if args.case == 3:
        cp = _last_by_pid(found["cp_attention"])
        stats = {pid: {"pcp_rank": int(m[1]), "dcp_rank": int(m[2]), "stats": json.loads(m[3].replace("'", '"'))}
                 for pid, m in cp.items()}
        ok = None if not stats else (len({s["dcp_rank"] for s in stats.values()}) >= 2 and all(
            s["stats"]["prefill_calls"] > 0 and s["stats"]["kv_rows_written"] > 0 for s in stats.values()))
        item("pcp_dcp_attention", ok, stats)
    urls = sorted(set(found["wandb_url"]))
    item("wandb_run_url", None if not urls else True, urls)
    passed = all(entry["status"] == "pass" for entry in items.values())
    receipt = {"case": args.case, "run_dir": str(args.run_dir), "passed": passed, "items": items,
               "negative_controls": "not_run (user policy)", "full_rows_retained": False}
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({"passed": passed, **{k: v["status"] for k, v in items.items()}}, indent=2))


if __name__ == "__main__":
    main()
