"""Find a Hugging Face model already in the local HF cache, for tests that check a real released
checkpoint when one is present. Never downloads: the lookup is ``local_files_only``, so it honors
``HF_HOME`` / ``HF_HUB_CACHE`` and returns ``None`` on any machine that has not fetched the model.
"""

from __future__ import annotations

import os


def cached_snapshot(repo_id: str, revision: str | None = None, *, require: str = "config.json") -> str | None:
    """The local snapshot directory of ``repo_id`` (at ``revision``, default the cached ``main``) if
    it holds ``require``; ``None`` when ``huggingface_hub`` is missing or the model is not cached."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        path = snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 -- not cached (LocalEntryNotFoundError) or an unreadable cache
        return None
    return path if os.path.isfile(os.path.join(path, require)) else None


def require_snapshot(repo_id: str, revision: str | None = None, *, require: str = "config.json") -> str:
    """``cached_snapshot`` or ``pytest.skip`` naming the model and how to provide it."""
    path = cached_snapshot(repo_id, revision, require=require)
    if path is None:
        import pytest

        rev = f" --revision {revision}" if revision else ""
        pytest.skip(
            f"{repo_id} {require} is not in the local Hugging Face cache (HF_HOME / HF_HUB_CACHE); "
            f"`hf download {repo_id} {require}{rev}` to run this check"
        )
    return path
