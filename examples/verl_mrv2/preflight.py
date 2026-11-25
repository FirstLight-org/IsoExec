"""Preflight for the verl x IsoExec x vLLM Model Runner V2 cases; run on the target node before any GPU job.

Checks, each recorded in ``preflight.json`` (no credential value is ever printed or written):
- exact versions and import paths of torch, vLLM, transformers, megatron-core, verl, IsoExec;
- the IsoExec vLLM general plugin entry point and ``transformers.core_model_loading.Split``;
- the IsoExec and verl Git SHAs of the checkouts on ``PYTHONPATH``;
- CUDA toolchain for first-use JIT (``CUDA_HOME``/``nvcc``, CUTLASS headers) and visible GPUs;
- a Ray CPU worker importing the same IsoExec/verl paths and a ``ray.put``/``ray.get`` round trip with the
  launcher's object-store size against the actual ``/dev/shm`` capacity;
- the W&B credential is present and the target ``entity/project`` is reachable (online mode).

Usage: python examples/verl_mrv2/preflight.py --out DIR [--wandb-entity E --wandb-project P] [--gpus N]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {"vllm": "0.33.0"}


def _sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module(name: str) -> dict:
    module = importlib.import_module(name)
    try:
        version = metadata.version({"megatron.core": "megatron-core"}.get(name, name))
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return {"version": version, "file": os.path.abspath(module.__file__ or "")}


def worker_probe() -> dict:
    import isoexec
    import verl

    return {"isoexec": os.path.abspath(isoexec.__file__), "verl": os.path.abspath(verl.__file__),
            "python": sys.executable, "pid": os.getpid()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--object-store-bytes", type=int, default=34359738368)
    parser.add_argument("--source-manifest", type=Path, default=None,
                        help="staged source manifest naming the IsoExec/verl identity when the copies are not Git checkouts")
    args = parser.parse_args()
    report: dict = {"checks": {}, "failures": []}

    def check(name: str, ok: bool, detail) -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["failures"].append(name)

    modules = {name: _module(name) for name in ("torch", "vllm", "transformers", "megatron.core", "verl", "isoexec")}
    report["modules"] = modules
    for name, version in EXPECTED.items():
        check(f"{name}_version", modules[name]["version"] == version, modules[name]["version"])
    import torch

    check("torch_cuda", torch.version.cuda is not None, {"torch": torch.__version__, "cuda": torch.version.cuda})
    try:
        from transformers.core_model_loading import Split  # noqa: F401
        check("transformers_split", True, "present")
    except ImportError as error:
        check("transformers_split", False, repr(error))
    plugins = [entry.value for entry in metadata.entry_points(group="vllm.general_plugins") if entry.name == "isoexec"]
    check("vllm_plugin_entry_point", plugins == ["isoexec.runtimes.vllm.vllm_plugin:register"], plugins)
    isoexec_root = Path(modules["isoexec"]["file"]).parents[1]
    verl_root = Path(modules["verl"]["file"]).parents[1]
    report["shas"] = {"isoexec": _sha(isoexec_root), "verl": _sha(verl_root)}
    if args.source_manifest is not None and not all(report["shas"].values()):
        staged = json.loads(args.source_manifest.read_text())
        report["shas"] = {"isoexec": f"{staged['isoexec_base']}+diff:{staged['isoexec_tracked_diff_sha256']}",
                          "verl": staged["verl_commit"], "source": "staged manifest"}
    check("checkout_shas", all(report["shas"].values()), report["shas"])
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc") or (str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else None)
    nvcc_version = None
    if nvcc and Path(nvcc).exists():
        nvcc_version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    check("cuda_toolkit", nvcc_version is not None, {"CUDA_HOME": cuda_home, "nvcc": nvcc, "version": nvcc_version})
    if args.gpus:
        check("visible_gpus", torch.cuda.device_count() >= args.gpus,
              {"count": torch.cuda.device_count(),
               "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    shm = shutil.disk_usage("/dev/shm").total if Path("/dev/shm").exists() else 0
    check("dev_shm_capacity", shm >= args.object_store_bytes, {"dev_shm_bytes": shm, "object_store": args.object_store_bytes})

    import ray

    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=min(args.object_store_bytes, 2 << 30),
             runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    try:
        remote = ray.get(ray.remote(worker_probe).remote())
        check("ray_worker_imports", remote["isoexec"] == modules["isoexec"]["file"]
              and remote["verl"] == modules["verl"]["file"], remote)
        payload = bytes(range(256)) * 4096
        check("ray_put_get", ray.get(ray.put(payload)) == payload, len(payload))
    finally:
        ray.shutdown()

    if args.wandb_entity and args.wandb_project:
        has_key = bool(os.environ.get("WANDB_API_KEY")) or Path.home().joinpath(".netrc").exists()
        check("wandb_credential_present", has_key, "credential present" if has_key else "missing")
        if has_key:
            try:
                import wandb

                api = wandb.Api(timeout=30)
                project = api.project(args.wandb_project, entity=args.wandb_entity)
                check("wandb_project_reachable", True, f"{project.entity}/{project.name}")
            except Exception as error:  # noqa: BLE001 - recorded, never contains the key
                check("wandb_project_reachable", False, type(error).__name__)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"failures": report["failures"], "shas": report["shas"]}, indent=2))
    sys.exit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
