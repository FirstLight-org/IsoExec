"""Resolve the native model source and its checkpoint artifacts once in the driver; workers read the
same JSON. Imports stay free of runtime and CUDA initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from isoexec.compiler.source import ModelSource, _json_config

_RESOLVED_CONFIG = "_resolved_source_config"


def validate_model_config(selected: object) -> Mapping[str, Any] | None:
    if selected is None:
        return None
    if not isinstance(selected, Mapping) or set(selected) - {"model_package", "revision", "raw_config", "trainer_placement", "inference_placement", "trainer_capture_max_tokens", "full_logprob_comparison", _RESOLVED_CONFIG}:
        raise ValueError("isoexec requires model_package; optional keys are revision, raw_config, trainer_placement, "
                         "inference_placement, trainer_capture_max_tokens and full_logprob_comparison")
    package = selected.get("model_package")
    if not isinstance(package, str) or not package:
        raise ValueError("isoexec.model_package must be a nonempty importable Python package name")
    ModelSource(package)
    revision = selected.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("isoexec.revision must be a nonempty HF revision string")
    return selected


def _sections(trainer: Any) -> tuple[str, ...]:
    algorithm = trainer.algorithm
    use_ref = getattr(algorithm, "use_kl_loss", False) or getattr(algorithm, "use_kl_in_reward", False)
    return ("policy", "ref") if use_ref else ("policy",)


def _request_key(selected: Mapping[str, Any], paths: Mapping[str, str],
                 base_digests: Mapping[str, str] | None = None,
                 artifacts: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    overrides = selected.get("raw_config", {})
    if not isinstance(overrides, Mapping) or any(not isinstance(key, str) for key in overrides):
        raise ValueError("isoexec.raw_config must be a JSON mapping with string keys")
    patch = _json_config(dict(overrides))
    try:
        encoded = json.dumps({"source": ModelSource(package=selected["model_package"]).to_dict(),
                              "policy_revision": selected.get("revision"),
                              "patch": json.dumps(patch, allow_nan=False, separators=(",", ":")),
                              "paths": dict(paths), "base_sha256": dict(base_digests or {}),
                              "artifacts": dict(artifacts or {})},
                             allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("isoexec.raw_config must contain finite JSON-compatible values") from exc
    # Mapping order is author-visible; do not sort nested config keys.
    return hashlib.sha256(encoded.encode()).hexdigest()


def _artifact_records(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) - {"policy", "ref"}:
        raise ValueError("Checkpoint origin records must describe policy/ref artifacts")
    result = {}
    for role, record in value.items():
        if (not isinstance(record, Mapping) or set(record) != {"repo_id", "revision", "requested_revision", "path"}
                or not isinstance(record["repo_id"], str) or not record["repo_id"]
                or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
                or not isinstance(record["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is None
                or (record["requested_revision"] is not None and
                    (not isinstance(record["requested_revision"], str) or not record["requested_revision"]))):
            raise ValueError("Checkpoint origin record lacks its immutable HF request/path identity")
        result[role] = dict(record)
    return result


def _local_artifact(checkpoint: str) -> Path:
    """Reuse the production reader's file/format admission before starting workers."""
    from isoexec.compiler.weights import resolve_checkpoint
    from isoexec.storage import CheckpointStore

    path = resolve_checkpoint(checkpoint)
    config = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Checkpoint artifact requires config.json beside its safetensors: {path}")
    store = CheckpointStore(str(path))
    try:
        if not store.keys():
            raise ValueError(f"Checkpoint artifact contains no tensor keys: {path}")
    finally:
        store.close()
    return path


def prepare_checkpoints(cfg: Any) -> None:
    """Resolve artifacts and effective model JSON once in the driver.

    The cache/snapshot path must be mounted at the same path on every worker.
    Workers receive immutable local paths and one resolved source-config record
    inside the existing JSON mapping; they never repeat configuration merging.
    ``revision`` selects the policy Hub artifact. A reference using that same
    repository shares its pin; other reference artifacts are independently pinned.
    """
    selected = validate_model_config(getattr(cfg, "isoexec", None))
    trainer = cfg.trainer
    if selected is None:
        return
    _request_key(selected, {})  # Refuse invalid patches before resolving artifacts.
    from isoexec.compiler.weights import local_checkpoint_path
    previous = selected.get(_RESOLVED_CONFIG)
    saved = previous.get("artifacts", {}) if isinstance(previous, Mapping) else {}
    records = _artifact_records(saved if saved else getattr(cfg, "_isoexec_checkpoint_artifacts", {}))
    roles = _sections(trainer)
    previous_roles = previous.get("active_roles", tuple(records)) if isinstance(previous, Mapping) else tuple(records)
    if (not isinstance(previous_roles, (list, tuple))
            or any(not isinstance(name, str) or name not in ("policy", "ref") for name in previous_roles)
            or len(set(previous_roles)) != len(previous_roles)):
        raise ValueError("Checkpoint active-role provenance is invalid")
    origins = {}
    reused = {}
    for name in roles:
        checkpoint = getattr(trainer, name).model.path
        record = records.get(name)
        reused[name] = record is not None and str(local_checkpoint_path(checkpoint)) == record["path"]
        origins[name] = record["repo_id"] if reused[name] and record is not None else checkpoint
    # Inactive roles retain their JSON provenance without touching their files.
    # When reactivated, an explicitly changed path discards its old authority.
    records = {name: record for name, record in records.items() if name not in roles or reused[name]}
    paths: dict[str, str] = {}
    resolved: dict[tuple[str, str | None], tuple[str, dict[str, Any]]] = {}
    # A role enabled later must inherit an existing immutable pin instead of
    # asking a mutable Hub branch again. Refuse ambiguous retained authorities
    # before resolving or publishing any new paths.
    for name in roles:
        record = records.get(name)
        if record is None:
            continue
        if name not in previous_roles and origins[name] == origins["policy"]:
            # A reactivated reference's retained pin is historical. The active
            # policy owns this repository request, even after a branch cycles
            # A -> B -> A and A has moved. Active conflicting pins still refuse.
            continue
        request = record["repo_id"], record["requested_revision"]
        existing = resolved.get(request)
        if existing is not None and existing[1]["revision"] != record["revision"]:
            raise ValueError("Retained HF artifacts disagree on the commit for the same repository/revision request")
        resolved.setdefault(request, (record["path"], record))
    for name in roles:
        section = getattr(trainer, name)
        checkpoint = section.model.path
        record = records.get(name)
        requested = (selected.get("revision") if name == "policy" or origins[name] == origins["policy"]
                     else record["requested_revision"] if record is not None else None)
        follows_policy = name != "policy" and name not in previous_roles and origins[name] == origins["policy"]
        if record is not None and record["requested_revision"] == requested and not follows_policy:
            paths[name] = str(_local_artifact(checkpoint))
            continue
        path = Path(checkpoint).expanduser()
        if record is None and path.exists():
            paths[name] = str(_local_artifact(str(path)))
            continue
        if record is None and (path.is_absolute() or checkpoint.startswith((".", "~")) or checkpoint.endswith((".safetensors", ".bin", ".pt"))):
            raise FileNotFoundError(f"Checkpoint artifact is unavailable: {checkpoint}; mount the resolved snapshot on every worker")
        origin = origins[name]
        request = origin, requested
        if request not in resolved:
            from huggingface_hub import HfApi, snapshot_download

            commit = HfApi().model_info(repo_id=origin, revision=requested).sha
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise ValueError("HF model metadata did not provide an immutable commit SHA")
            snapshot = Path(snapshot_download(repo_id=origin, revision=commit, allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.py", "*.jinja", "tokenizer*",
            ])).resolve()
            if snapshot.name != commit or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
                raise FileNotFoundError("HF snapshot is missing its pinned config.json")
            _local_artifact(str(snapshot))
            # Fully qualified auto_map references request another Hub artifact,
            # even when loading this local snapshot. They need their own pin.
            for config_name in ("config.json", "tokenizer_config.json"):
                config_file = snapshot / config_name
                if config_file.is_file():
                    values = json.loads(config_file.read_text()).get("auto_map", {})
                    if "--" in json.dumps(values):
                        raise ValueError("HF artifact has external auto_map code; provide a local artifact with pinned code dependencies")
            record = {"repo_id": origin, "revision": commit, "requested_revision": requested, "path": str(snapshot)}
            resolved[request] = str(snapshot), record
        paths[name], pin = resolved[request]
        records[name] = dict(pin)

    directories = {configuration_path(path) for path in paths.values()}
    base = {directory: _json_config(json.loads((Path(directory) / "config.json").read_text(encoding="utf-8")))
            for directory in sorted(directories)}
    base_digests = {directory: hashlib.sha256(json.dumps(raw, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
                    for directory, raw in base.items()}
    request = _request_key(selected, paths, base_digests, records)
    if (isinstance(previous, Mapping) and previous.get("request") == request
            and isinstance(previous.get("configs"), dict) and set(previous["configs"]) == directories):
        effective = {directory: _json_config(previous["configs"][directory]) for directory in sorted(directories)}
    else:
        source = ModelSource(package=selected["model_package"])
        effective = {directory: source.apply_overrides(raw, selected.get("raw_config", {})) for directory, raw in base.items()}
    json.dumps(effective, allow_nan=False)
    # Publish no partial paths/configs when resolution or JSON validation fails.
    for name, path in paths.items():
        getattr(trainer, name).model.path = path
    cfg._isoexec_checkpoint_artifacts = records
    cfg.isoexec = {**selected, _RESOLVED_CONFIG: {"request": request, "base_sha256": base_digests,
                                                        "configs": effective, "artifacts": records,
                                                        "active_roles": list(roles)}}


def configuration_path(checkpoint: str) -> str:
    """Configuration/tokenizer directory beside an exact local weight file."""
    from isoexec.compiler.weights import local_checkpoint_path

    path = Path(checkpoint).expanduser()
    if not path.exists():
        return checkpoint
    path = local_checkpoint_path(path)
    return str(path.parent) if path.is_file() else str(path)


def source_config(isoexec: object, trainer: Any, checkpoint: str) -> tuple[ModelSource, dict[str, Any]]:
    """Read a private copy of driver-resolved JSON without importing author code."""
    selected = validate_model_config(isoexec)
    if selected is None:
        raise ValueError(
            "IsoExec requires isoexec.model_package, for example 'isoexec.models.qwen3_5'"
        )
    source = ModelSource(package=selected["model_package"])
    path = Path(checkpoint).expanduser()
    if not path.exists() and (path.is_absolute() or checkpoint.startswith((".", "~"))):
        raise FileNotFoundError(f"Checkpoint configuration path does not exist: {checkpoint}")
    if not path.exists():
        raise ValueError("Hub artifacts must be resolved once by the driver before source_config; "
                         "use prepare_checkpoints() and mount its immutable snapshot path on every worker")
    from isoexec.compiler.weights import local_checkpoint_path

    path = local_checkpoint_path(path)
    paths = {name: str(local_checkpoint_path(getattr(trainer, name).model.path)) for name in _sections(trainer)}
    prepared = selected.get(_RESOLVED_CONFIG)
    directories = {configuration_path(value) for value in paths.values()}
    if (not isinstance(prepared, Mapping)
            or not {"request", "base_sha256", "configs", "artifacts"}.issubset(prepared)
            or set(prepared) - {"request", "base_sha256", "configs", "artifacts", "active_roles"}
            or ("active_roles" in prepared and prepared["active_roles"] != list(_sections(trainer)))
            or not isinstance(prepared["base_sha256"], dict) or set(prepared["base_sha256"]) != directories
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in prepared["base_sha256"].values())
            or prepared["request"] != _request_key(selected, paths, prepared["base_sha256"], _artifact_records(prepared["artifacts"]))
            or str(path) not in paths.values()
            or not isinstance(prepared["configs"], dict) or set(prepared["configs"]) != directories):
        raise ValueError("Model source configuration is missing or stale; run prepare_checkpoints() in the driver before starting workers")
    raw = prepared["configs"][configuration_path(str(path))]
    return source, _json_config(raw)


__all__ = [
    "source_config",
    "prepare_checkpoints",
    "configuration_path",
]
