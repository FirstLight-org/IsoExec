"""Verify both vendored kernel trees against their own PROVENANCE.json without importing a runtime.

The trees were qualified for determinism when they were vendored. This checks only OUR copy: every
vendored file still hashes to its recorded ``vendored_sha256``. The recorded upstream (distribution,
version, commit, per-file ``upstream_sha256``) is citation and is not compared with an installed
package."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field


@dataclass
class TreeReport:
    """What the check found. ``ok`` is the only thing a caller has to look at."""

    tree: str
    upstream: dict
    ok: bool = True
    tampered: list[str] = field(default_factory=list)  # vendored file != recorded hash
    checked: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"{self.tree}: {len(self.checked)} file(s) match their recorded hashes (vendored from {self.upstream_label()})"
        return f"{self.tree}: HAND-EDITED: {', '.join(sorted(self.tampered))}"

    def upstream_label(self) -> str:
        u = self.upstream
        return f"{u.get('distribution')} {u.get('version')} (commit {u.get('commit')})"


def load(tree_dir: str) -> dict:
    with open(os.path.join(tree_dir, "PROVENANCE.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_vendored_hashes(tree_dir: str) -> TreeReport:
    doc = load(tree_dir)
    rep = TreeReport(tree=doc["provenance_label"].rsplit("/", 1)[0], upstream=doc["upstream"])
    for entry in doc["files"]:
        path = os.path.join(tree_dir, entry["file"])
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        rep.checked.append(entry["file"])
        if _sha(text) != entry["vendored_sha256"]:
            rep.ok = False
            rep.tampered.append(entry["file"])
    return rep


