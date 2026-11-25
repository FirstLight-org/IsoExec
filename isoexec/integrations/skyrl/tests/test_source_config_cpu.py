"""Explicit model sources cross the actual SkyRL registration seam as JSON."""

import importlib
from copy import deepcopy
from collections.abc import Iterator
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from isoexec.integrations.skyrl import model_folder
from isoexec.integrations.skyrl.config import configure_isoexec_vllm_engine, resolve
from isoexec.integrations.skyrl.tests.test_config_cpu import config


def _artifact_weights(folder: Path, name: str = "model.safetensors") -> Path:
    """Genuine safetensors structure only; no model/kernel qualification claim."""
    path = folder / name
    save_file({"probe": torch.tensor([1.0])}, str(path))
    return path


@pytest.fixture
def config_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    package = tmp_path / "override_author"
    package.mkdir()
    (package / "model.py").write_text(
        "from isoexec.compiler.tests.test_capture_cpu import Model as Base\n"
        "class Model(Base):\n"
        "    def __init__(self, cfg):\n"
        "        super().__init__()\n"
        "        self.scale = cfg.text['scale'] * cfg.text.get('base_factor', 1)\n")
    (package / "weight.py").write_text(
        "def checkpoint_conversion_mapping(config): return []\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield package
    for name in tuple(sys.modules):
        if name == package.name or name.startswith(package.name + "."):
            sys.modules.pop(name)


def test_real_skyrl_cli_accepts_source_and_separate_placement_policy():
    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig.from_cli_overrides([
        "isoexec.model_package=isoexec.models.qwen3_5",
        "isoexec.trainer_placement.leaves=16",
        "isoexec.inference_placement.leaves=8",
        "trainer.policy.megatron_config.pipeline_model_parallel_size=2",
        "generator.inference_engine.pipeline_parallel_size=4",
    ])
    assert cfg.isoexec["model_package"] == "isoexec.models.qwen3_5"
    assert cfg.isoexec["trainer_placement"] == {"leaves": 16}
    assert cfg.isoexec["inference_placement"] == {"leaves": 8}
    assert cfg.trainer.policy.megatron_config.pipeline_model_parallel_size == 2
    assert cfg.generator.inference_engine.pipeline_parallel_size == 4


def test_source_config_resolves_local_or_hf_config_without_family_routing(tmp_path: Path,
                                                                       config_package: Path):
    raw = {"custom_author_field": 3, "nested": {"left": 1, "right": 2}}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    _artifact_weights(tmp_path)
    trainer = SimpleNamespace(
        isoexec={
            "model_package": config_package.name,
            "raw_config": {"custom_author_field": 7, "nested": {"right": 3}},
            "trainer_placement": {"leaves": 16},
            "inference_placement": {"leaves": 8, "leaf_dtype": "bf16"},
        },
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path=str(tmp_path))), algorithm=SimpleNamespace()
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    local = model_folder.source_config(trainer.isoexec, trainer.trainer, str(tmp_path))
    assert local is not None and local[0].package == config_package.name
    assert local[1] == {"custom_author_field": 7, "nested": {"left": 1, "right": 3}}
    weight_file = tmp_path / "model.safetensors"
    _artifact_weights(tmp_path)
    trainer.trainer.policy.model.path = str(weight_file)
    model_folder.prepare_checkpoints(trainer)
    assert model_folder.source_config(trainer.isoexec, trainer.trainer, str(weight_file)) == local
    assert model_folder.configuration_path(str(weight_file)) == str(tmp_path)
    with pytest.raises(ValueError, match="resolved once by the driver"):
        model_folder.source_config(trainer.isoexec, trainer.trainer, 'custom-author/checkpoint')


def test_hf_artifacts_resolve_one_commit_for_enabled_roles_and_never_requery_workers(tmp_path: Path,
                                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                                   config_package: Path):
    import torch
    from safetensors.torch import save_file, load_file
    from isoexec.compiler.weights import resolve_checkpoint
    commit = "a" * 40
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"commit_marker": "A"}')
    (snapshot / "tokenizer_config.json").write_text('{"commit_marker": "A"}')
    save_file({"weight": torch.tensor([1.0])}, str(snapshot / "model.safetensors"))
    calls = []
    class API:
        def model_info(self, **kwargs):
            calls.append(("info", kwargs))
            return SimpleNamespace(sha=commit if len(calls) == 1 else "b" * 40)
    def download(**kwargs):
        calls.append(("snapshot", kwargs))
        assert kwargs["revision"] == commit  # The branch may now refer to B.
        return str(snapshot)
    monkeypatch.setattr("huggingface_hub.HfApi", API)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name, "revision": "moving-branch"},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            ref=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            algorithm=SimpleNamespace(use_kl_loss=True),
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    model_folder.prepare_checkpoints(trainer)
    assert len(calls) == 2 and calls[0] == ("info", {"repo_id": "author/repo", "revision": "moving-branch"})
    assert trainer.trainer.policy.model.path == trainer.trainer.ref.model.path == str(snapshot)
    captured = model_folder.source_config(trainer.isoexec, trainer.trainer, trainer.trainer.policy.model.path)
    assert captured is not None and captured[1]["commit_marker"] == "A"
    assert resolve_checkpoint(trainer.trainer.policy.model.path) == snapshot
    assert load_file(str(snapshot / "model.safetensors"))["weight"].item() == 1
    assert len(calls) == 2
    with pytest.raises(FileNotFoundError, match="does not exist"):
        model_folder.source_config(trainer.isoexec, trainer.trainer, str(tmp_path / 'unmounted-snapshot'))


def test_later_same_repository_reference_inherits_pin_and_conflicting_authorities_refuse(
    tmp_path: Path, config_package: Path, monkeypatch: pytest.MonkeyPatch,
):
    snapshots = {}
    for letter in ('a', 'b'):
        path = tmp_path / 'snapshots' / (letter * 40)
        path.mkdir(parents=True)
        (path / 'config.json').write_text(json.dumps({'commit_marker': letter}))
        _artifact_weights(path)
        snapshots[letter * 40] = path
    calls = []
    def info(**kwargs):
        calls.append(('info', kwargs))
        return SimpleNamespace(sha=('a' if len(calls) == 1 else 'b') * 40)
    def download(**kwargs):
        calls.append(('snapshot', kwargs))
        return str(snapshots[kwargs['revision']])
    monkeypatch.setattr('huggingface_hub.HfApi', lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name, "revision": "main"},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            ref=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            algorithm=SimpleNamespace(use_kl_loss=False),
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    trainer.trainer.algorithm.use_kl_loss = True
    model_folder.prepare_checkpoints(trainer)
    assert len(calls) == 2  # Mutable main would now report B; it must not be queried again.
    assert trainer.trainer.policy.model.path == trainer.trainer.ref.model.path == str(snapshots['a' * 40])
    records = trainer.isoexec['_resolved_source_config']['artifacts']
    assert records['ref']['revision'] == records['policy']['revision'] == 'a' * 40
    records['ref']['revision'], records['ref']['path'] = 'b' * 40, str(snapshots['b' * 40])
    trainer.trainer.ref.model.path = records['ref']['path']
    before = deepcopy(trainer.isoexec)
    paths = trainer.trainer.policy.model.path, trainer.trainer.ref.model.path
    with pytest.raises(ValueError, match='disagree on the commit'):
        model_folder.prepare_checkpoints(trainer)
    assert trainer.isoexec == before and (trainer.trainer.policy.model.path, trainer.trainer.ref.model.path) == paths
    assert len(calls) == 2


@pytest.mark.parametrize('replace_reference', [False, True])
def test_inactive_reference_provenance_survives_revision_change_and_json_without_file_reads(
    tmp_path: Path, config_package: Path, monkeypatch: pytest.MonkeyPatch, replace_reference: bool,
):
    snapshots = {}
    for letter in ('a', 'b', 'c'):
        directory = tmp_path / 'snapshots' / (letter * 40)
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text(json.dumps({'commit_marker': letter}))
        _artifact_weights(directory)
        snapshots[letter] = directory
    calls = []
    def info(**kwargs):
        calls.append(('info', kwargs))
        return SimpleNamespace(sha=kwargs['revision'].lower() * 40)
    def download(**kwargs):
        calls.append(('snapshot', kwargs))
        return str(snapshots[kwargs['revision'][0]])
    monkeypatch.setattr('huggingface_hub.HfApi', lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name, "revision": "A"},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            ref=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            algorithm=SimpleNamespace(use_kl_loss=True),
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    assert trainer.trainer.policy.model.path == trainer.trainer.ref.model.path == str(snapshots['a'])
    trainer.trainer.algorithm.use_kl_loss = False
    model_folder.prepare_checkpoints(trainer)
    trainer.isoexec = json.loads(json.dumps(trainer.isoexec))
    delattr(trainer, '_isoexec_checkpoint_artifacts')  # The JSON record is the transport authority.
    trainer.isoexec['revision'] = 'B'
    model_folder.prepare_checkpoints(trainer)
    assert trainer.trainer.policy.model.path == str(snapshots['b']) and trainer.trainer.ref.model.path == str(snapshots['a'])
    retained = trainer.isoexec['_resolved_source_config']['artifacts']['ref']
    assert retained['repo_id'] == 'author/repo' and retained['requested_revision'] == 'A'
    # An inactive artifact may disappear. Preparation/worker config must never
    # open it, nor use it as an active repository/revision resolution seed.
    (snapshots['a'] / 'model.safetensors').unlink()
    model_folder.prepare_checkpoints(trainer)
    configured = model_folder.source_config(trainer.isoexec, trainer.trainer, trainer.trainer.policy.model.path)
    assert configured is not None and configured[1]['commit_marker'] == 'b'
    trainer.isoexec = json.loads(json.dumps(trainer.isoexec))
    if replace_reference:
        trainer.trainer.ref.model.path = str(snapshots['c'])
    trainer.trainer.algorithm.use_kl_loss = True
    model_folder.prepare_checkpoints(trainer)
    expected = snapshots['c' if replace_reference else 'b']
    assert trainer.trainer.ref.model.path == str(expected)
    records = trainer.isoexec['_resolved_source_config']['artifacts']
    if replace_reference:
        assert 'ref' not in records  # User's new independent local selection owns its provenance.
    else:
        assert records['ref'] == records['policy'] and records['ref']['revision'] == 'b' * 40
    assert [kwargs for kind, kwargs in calls if kind == 'info'] == [
        {'repo_id': 'author/repo', 'revision': 'A'}, {'repo_id': 'author/repo', 'revision': 'B'}]


@pytest.mark.parametrize('moving_branch', [False, True])
def test_reference_reactivation_follows_current_policy_after_revision_round_trip(
    tmp_path: Path, config_package: Path, monkeypatch: pytest.MonkeyPatch, moving_branch: bool,
):
    snapshots = {}
    for letter in ('a', 'b', 'c'):
        path = tmp_path / 'snapshots' / (letter * 40)
        path.mkdir(parents=True)
        (path / 'config.json').write_text(json.dumps({'commit_marker': letter}))
        _artifact_weights(path)
        snapshots[letter] = path
    commits = {'A': 'a', 'B': 'b'}
    calls = []
    def info(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sha=commits[kwargs['revision']] * 40)
    monkeypatch.setattr('huggingface_hub.HfApi', lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda **kwargs: str(snapshots[kwargs['revision'][0]]))
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name, "revision": "A"},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            ref=SimpleNamespace(model=SimpleNamespace(path="author/repo")),
            algorithm=SimpleNamespace(use_kl_loss=True),
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    trainer.trainer.algorithm.use_kl_loss = False
    model_folder.prepare_checkpoints(trainer)
    trainer.isoexec['revision'] = 'B'
    model_folder.prepare_checkpoints(trainer)
    commits['A'] = 'c' if moving_branch else 'a'
    trainer.isoexec['revision'] = 'A'
    model_folder.prepare_checkpoints(trainer)
    historical = trainer.isoexec['_resolved_source_config']['artifacts']['ref']
    assert historical['revision'] == 'a' * 40
    trainer.isoexec = json.loads(json.dumps(trainer.isoexec))
    trainer.trainer.algorithm.use_kl_loss = True
    model_folder.prepare_checkpoints(trainer)
    current = snapshots[commits['A']]
    assert trainer.trainer.policy.model.path == trainer.trainer.ref.model.path == str(current)
    records = trainer.isoexec['_resolved_source_config']['artifacts']
    assert records['policy'] == records['ref']
    assert len(calls) == 3  # Reactivation reuses the active pin; no fourth mutable lookup.


def test_exact_symlink_weight_keeps_selected_metadata_root_and_real_bytes(
    tmp_path: Path, config_package: Path,
):
    from safetensors.torch import load_file

    from isoexec.compiler.weights import resolve_checkpoint
    weights = {"tokens": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    blobs = tmp_path / 'blob-data'
    blobs.mkdir()
    blob = blobs / ('d' * 64)
    save_file(weights, str(blob))
    metadata = tmp_path / 'selected-metadata'
    metadata.mkdir()
    (metadata / 'config.json').write_text('{"commit_marker": "selected"}')
    selected = metadata / 'model.safetensors'
    selected.symlink_to(blob)
    assert selected.resolve() == blob and resolve_checkpoint(selected) == selected
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path=str(selected))), algorithm=SimpleNamespace()
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    assert trainer.trainer.policy.model.path == str(selected)
    assert model_folder.configuration_path(str(selected)) == str(metadata)
    configured = model_folder.source_config(trainer.isoexec, trainer.trainer, str(selected))
    assert configured is not None and configured[1]['commit_marker'] == 'selected'
    worker = deepcopy(trainer)
    worker.isoexec = json.loads(json.dumps(trainer.isoexec))
    assert model_folder.source_config(worker.isoexec, worker.trainer, str(selected)) == configured
    # Readers follow the selected link to the real bytes.
    assert torch.equal(load_file(str(selected))['tokens'], weights['tokens'])


@pytest.mark.parametrize("shared_ref", [True, False])
def test_changed_policy_revision_preserves_hub_origin_across_json_and_publishes_atomically(
    tmp_path: Path, config_package: Path, monkeypatch: pytest.MonkeyPatch, shared_ref: bool,
):
    snapshots = {}
    for marker in ("a", "b", "c"):
        directory = tmp_path / "snapshots" / (marker * 40)
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"commit_marker": marker}))
        _artifact_weights(directory)
        snapshots[marker * 40] = directory
    calls = []
    def info(**kwargs):
        calls.append(("info", kwargs))
        sha = "c" * 40 if kwargs["repo_id"] == "author/reference" else kwargs["revision"].lower() * 40
        return SimpleNamespace(sha=sha)
    def download(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(snapshots[kwargs["revision"]])
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name, "revision": "A"},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="author/policy")),
            ref=SimpleNamespace(
                model=SimpleNamespace(path="author/policy" if shared_ref else "author/reference")
            ),
            algorithm=SimpleNamespace(use_kl_loss=True),
        ),
    )
    model_folder.prepare_checkpoints(trainer)
    original_calls = len(calls)
    # The driver-origin record must survive the actual JSON boundary rather
    # than depending on an unregistered attribute of the old Python object.
    copied = SimpleNamespace(
        isoexec=json.loads(json.dumps(trainer.isoexec)),
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path=trainer.trainer.policy.model.path)),
            ref=SimpleNamespace(model=SimpleNamespace(path=trainer.trainer.ref.model.path)),
            algorithm=SimpleNamespace(use_kl_loss=True),
        ),
    )
    copied.isoexec["revision"] = "B"
    with pytest.raises(ValueError, match="missing or stale"):
        model_folder.source_config(copied.isoexec, copied.trainer, copied.trainer.policy.model.path)
    previous = deepcopy(copied.isoexec)
    previous_paths = copied.trainer.policy.model.path, copied.trainer.ref.model.path
    with monkeypatch.context() as failed:
        failed.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")))
        with pytest.raises(RuntimeError, match="download failed"):
            model_folder.prepare_checkpoints(copied)
    assert copied.isoexec == previous and (copied.trainer.policy.model.path, copied.trainer.ref.model.path) == previous_paths
    model_folder.prepare_checkpoints(copied)
    assert copied.trainer.policy.model.path == str(snapshots["b" * 40])
    assert copied.trainer.ref.model.path == str(snapshots[("b" if shared_ref else "c") * 40])
    assert calls[original_calls][1] == {"repo_id": "author/policy", "revision": "B"}
    record = copied.isoexec["_resolved_source_config"]["artifacts"]["policy"]
    assert record == {"repo_id": "author/policy", "requested_revision": "B", "revision": "b" * 40,
                      "path": str(snapshots["b" * 40])}
    selected = model_folder.source_config(copied.isoexec, copied.trainer, copied.trainer.policy.model.path)
    assert selected is not None and selected[1]["commit_marker"] == "b"
    complete_calls = len(calls)
    model_folder.prepare_checkpoints(copied)
    assert len(calls) == complete_calls


@pytest.mark.parametrize("kind,error,message", [
    ("bin", ValueError, "must use safetensors"),
    ("pt", ValueError, "must use safetensors"),
    ("config_only", FileNotFoundError, "no .safetensors"),
    ("invalid_safetensors", SafetensorError, "header"),
])
def test_driver_refuses_unreadable_local_artifacts_before_publishing(
    tmp_path: Path, config_package: Path, kind: str, error: type[Exception], message: str,
):
    directory = tmp_path / kind
    directory.mkdir()
    (directory / "config.json").write_text('{}')
    checkpoint = directory if kind == "config_only" else directory / (
        "weights.safetensors" if kind == "invalid_safetensors" else f"weights.{kind}")
    if kind != "config_only":
        checkpoint.write_bytes(b"invalid artifact")
    trainer = SimpleNamespace(
        isoexec={"model_package": config_package.name},
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path=str(checkpoint))), algorithm=SimpleNamespace()
        ),
    )
    before = deepcopy(trainer.isoexec)
    with pytest.raises(error, match=message):
        model_folder.prepare_checkpoints(trainer)
    assert trainer.isoexec == before and trainer.trainer.policy.model.path == str(checkpoint)
    assert not hasattr(trainer, "_isoexec_checkpoint_artifacts")


def test_nested_overrides_resolve_once_and_cross_worker_json_with_the_same_config(
    tmp_path: Path, config_package: Path, monkeypatch: pytest.MonkeyPatch,
):
    from isoexec.compiler.source import ModelSource, ModelSourceError

    calls = []
    apply_overrides = ModelSource.apply_overrides

    def counted(source, base, patch):
        calls.append(1)
        return apply_overrides(source, base, patch)

    monkeypatch.setattr(ModelSource, "apply_overrides", counted)
    base = {"model_type": "llama", "architectures": ["Model"],
            "text": {"hidden": 16, "vocab": 32, "scale": 0.5}}
    patch = {"text": {"scale": 0.25}}
    (tmp_path / "config.json").write_text(json.dumps(base))
    _artifact_weights(tmp_path)

    def trainer_for(scale):
        return SimpleNamespace(
            isoexec={
                "model_package": config_package.name,
                "raw_config": {"text": {"scale": scale}},
            },
            trainer=SimpleNamespace(
                policy=SimpleNamespace(model=SimpleNamespace(path=str(tmp_path))),
                ref=SimpleNamespace(model=SimpleNamespace(path=str(tmp_path))),
                algorithm=SimpleNamespace(use_kl_loss=True),
            ),
        )

    trainer = trainer_for(0.25)
    model_folder.prepare_checkpoints(trainer)
    model_folder.prepare_checkpoints(trainer)
    selected = model_folder.source_config(trainer.isoexec, trainer.trainer, str(tmp_path))
    assert selected is not None
    source, raw = selected
    assert len(calls) == 1 and raw == {**base, "text": {"hidden": 16, "vocab": 32, "scale": 0.25}}
    assert trainer.isoexec["raw_config"] == patch
    assert json.loads((tmp_path / "config.json").read_text()) == base
    assert source.to_dict() == {"package": config_package.name}

    expected = deepcopy(raw)
    raw["text"]["hidden"] = 99  # A consumer cannot mutate the prepared record.
    worker = trainer_for(0.25)
    worker.isoexec = json.loads(json.dumps(trainer.isoexec))
    # Workers consume the resolved JSON without repeating driver-side merging.
    worker_selected = model_folder.source_config(worker.isoexec, worker.trainer, str(tmp_path))
    assert worker_selected is not None and worker_selected[1]["text"]["hidden"] == 16
    assert worker_selected[1] == expected
    assert len(calls) == 1

    changed = trainer_for(0.5)
    model_folder.prepare_checkpoints(changed)
    changed_selected = model_folder.source_config(changed.isoexec, changed.trainer, str(tmp_path))
    assert changed_selected is not None and changed_selected[1] != expected
    assert len(calls) == 2  # A new driver request resolves once; workers never resolve it.
    changed_base = deepcopy(base)
    changed_base["text"]["base_factor"] = 2
    (tmp_path / "config.json").write_text(json.dumps(changed_base))
    model_folder.prepare_checkpoints(trainer)
    base_selected = model_folder.source_config(trainer.isoexec, trainer.trainer, str(tmp_path))
    assert base_selected is not None and base_selected[1]["text"]["base_factor"] == 2
    assert len(calls) == 3 and base_selected[1] != expected
    model_folder.prepare_checkpoints(trainer)
    assert len(calls) == 3
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps(base))
    for field in ("patch", "source", "path"):
        stale = deepcopy(worker)
        if field == "patch":
            stale.isoexec["raw_config"]["text"]["scale"] = 0.75
        elif field == "source":
            stale.isoexec["model_package"] += "_changed"
        else:
            stale.trainer.policy.model.path = str(other)
        with pytest.raises(ValueError, match="missing or stale.*prepare_checkpoints"):
            model_folder.source_config(stale.isoexec, stale.trainer, stale.trainer.policy.model.path)

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    for invalid in ({"nested": {1: "number", "1": "string"}}, {"value": float("nan")},
                    {"value": float("inf")}, {"value": object()}, {"value": (1, 2)}, cyclic):
        failed = deepcopy(worker)
        failed.isoexec["raw_config"] = invalid
        failed.trainer.policy.model.path = os.path.relpath(tmp_path)
        previous_path = failed.trainer.policy.model.path
        previous = deepcopy(failed.isoexec["_resolved_source_config"])
        with pytest.raises((ModelSourceError, ValueError), match="JSON"):
            model_folder.prepare_checkpoints(failed)
        assert failed.isoexec["_resolved_source_config"] == previous
        assert failed.isoexec["raw_config"] is invalid
        assert failed.trainer.policy.model.path == previous_path


def test_engine_registration_preserves_independent_native_parallelism_and_source(tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch):
    package = tmp_path / "skyrl_author"
    package.mkdir()
    (package / "model.py").write_text(
        "from isoexec.compiler.tests.test_capture_cpu import Model as Base\n"
        "class Model(Base):\n"
        "    def __init__(self, cfg): super().__init__()\n"
    )
    (package / "weight.py").write_text("def checkpoint_conversion_mapping(config): raise RuntimeError('registration must not load weights')\n")
    raw = {"model_type": "llama", "architectures": ["Model"], "custom_author_field": 3}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(raw))
    _artifact_weights(snapshot)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    cfg = config()
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.ckpt_interval = cfg.trainer.hf_save_interval = -1
    cfg.trainer.resume_mode = "none"
    cfg.trainer.max_prompt_length = 512
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.max_tokens_per_microbatch = -1
    cfg.generator.batched = True
    cfg.generator.sampling_params.max_generate_length = 1024
    cfg.trainer.policy.model.path = "author/repo"
    cfg.isoexec = {
        "model_package": package.name,
        "trainer_placement": {"leaves": 16},
        "inference_placement": {"leaves": 8, "leaf_dtype": "bf16"},
    }
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    engine = cfg.generator.inference_engine
    engine.pipeline_parallel_size, engine.tensor_parallel_size, engine.num_engines = 4, 2, 2
    engine.enforce_eager = True
    engine.max_num_batched_tokens = 32
    engine.engine_init_kwargs = {"compilation_config": {"pass_config": {"enable_sp": True}}}
    framework_extension = "skyrl.framework.WorkerExtension"
    args = SimpleNamespace(model="author/repo", tokenizer="author/repo", tensor_parallel_size=2, pipeline_parallel_size=4,
                           data_parallel_size=1, max_num_batched_tokens=32, additional_config=None,
                           compilation_config=engine.engine_init_kwargs["compilation_config"],
                           worker_extension_cls=framework_extension)
    hub_calls = []
    class HubApi:
        def model_info(self, **kwargs):
            hub_calls.append(kwargs)
            return SimpleNamespace(sha="a" * 40)
    monkeypatch.setattr("huggingface_hub.HfApi", HubApi)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs:
                        hub_calls.append(kwargs) or str(snapshot))
    # This CPU seam exercises configuration/registration; native dependency
    # admission is independently checked inside the actual GPU image.
    monkeypatch.setattr("isoexec.integrations.skyrl.config.dependencies.check", lambda **_kwargs: [])
    monkeypatch.setattr("isoexec.runtimes.vllm.registration.register_plugin", lambda: None)
    monkeypatch.setenv("VLLM_PLUGINS", "")
    try:
        # The driver resolves once (the Hub artifact, here); SkyRL then builds the server arguments
        # from the resolved policy path, which configure_isoexec_vllm_engine turns into the IsoExec engine's.
        resolve(cfg)
        args.model = args.tokenizer = cfg.trainer.policy.model.path
        configure_isoexec_vllm_engine(cfg, args)
        settings = json.loads(json.dumps(args.additional_config))["isoexec"]
        assert settings["model_source"] == {"package": cfg.isoexec["model_package"]}
        assert settings["raw_model_config"] == raw
        from isoexec.runtimes.vllm.registration import NATIVE_DENSE_ARCH

        assert args.hf_overrides["architectures"] == [NATIVE_DENSE_ARCH]
        assert args.load_format == "auto" and args.enforce_eager
        assert args.model == args.tokenizer == cfg.trainer.policy.model.path == str(snapshot)
        assert args.revision == args.tokenizer_revision == args.code_revision == "a" * 40
        assert len(hub_calls) == 2
        assert (args.tensor_parallel_size, args.pipeline_parallel_size, args.data_parallel_size) == (2, 4, 1)
        assert cfg.trainer.policy.megatron_config.pipeline_model_parallel_size == 2
        assert engine.num_engines == 2 and args.worker_extension_cls == framework_extension
        # A single-file source keeps exact weight selection while native config
        # and tokenizer APIs receive its containing directory.
        weight_file = snapshot / "selected.safetensors"
        blob_folder = tmp_path / "shared-bytes"
        blob_folder.mkdir()
        blob = _artifact_weights(blob_folder, "extensionless-blob")
        weight_file.symlink_to(blob)
        cfg.trainer.policy.model.path = str(weight_file)
        resolve(cfg)
        args.model, args.tokenizer = str(weight_file), None
        configure_isoexec_vllm_engine(cfg, args)
        assert args.model == args.tokenizer == str(snapshot)
        assert args.served_model_name == [str(weight_file)]
        assert args.additional_config["isoexec"]["checkpoint_path"] == str(weight_file)
        assert "checkpoint_artifact" not in args.additional_config["isoexec"]
        assert args.worker_extension_cls == framework_extension
        assert len(hub_calls) == 2
        with monkeypatch.context() as home_lookup:
            # Exercise tilde expansion without changing the process HOME.
            original_expand = os.path.expanduser
            home_lookup.setattr("os.path.expanduser", lambda path:
                                str(tmp_path) if path == "~" else
                                str(tmp_path) + path[1:] if path.startswith("~/") else original_expand(path))
            relative = "~/" + weight_file.relative_to(tmp_path).as_posix()
            cfg.trainer.policy.model.path = relative
            resolve(cfg)
            args.model = cfg.trainer.policy.model.path
            configure_isoexec_vllm_engine(cfg, args)
            assert cfg.trainer.policy.model.path == args.additional_config["isoexec"]["checkpoint_path"] == str(weight_file)
            assert args.model == str(snapshot)
    finally:
        for name in tuple(sys.modules):
            if name == package.name or name.startswith(package.name + "."):
                sys.modules.pop(name)
