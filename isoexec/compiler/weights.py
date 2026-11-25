"""Resolve Hugging Face checkpoint locations for model sources."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def local_checkpoint_path(checkpoint: str | Path) -> Path:
    """Canonicalize the directory, preserving an exact file's metadata selection.

    A selected safetensors symlink can point outside its config/tokenizer
    directory. Readers follow that link for bytes; resolving the final filename
    would instead discard the directory the caller selected for metadata.
    """
    path = Path(checkpoint).expanduser()
    return path.parent.resolve() / path.name if path.is_file() else path.resolve()


def resolve_checkpoint(checkpoint: str | Path, *, revision: str | None = None) -> Path:
    """Resolve an existing safetensors path or an HF repository to the same reader."""
    path = Path(checkpoint).expanduser()
    if path.exists():
        if path.is_dir() or path.suffix == ".safetensors":
            return local_checkpoint_path(path)
        raise ValueError(f"Checkpoint file must use safetensors: {path}")
    if isinstance(checkpoint, Path) or path.is_absolute() or str(checkpoint).startswith((".", "~")):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    if str(checkpoint).endswith((".safetensors", ".bin", ".pt")):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")
    try:
        hub = import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        if exc.name != "huggingface_hub":
            raise
        raise ImportError("HF checkpoint resolution requires huggingface_hub; local safetensors paths do not") from exc
    resolved = Path(
        hub.snapshot_download(
            repo_id=str(checkpoint),
            revision=revision,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    )
    if not resolved.is_dir():
        raise FileNotFoundError(f"HF checkpoint snapshot is not a directory: {resolved}")
    return resolved.resolve()
