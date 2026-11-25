#!/usr/bin/env python3
"""Fail on installed dependency metadata conflicts that environments/default.toml does not list."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def installed_conflicts(distributions=None) -> list[dict]:
    installed = {canonicalize_name(dist.metadata['Name']): dist for dist in
                 (metadata.distributions() if distributions is None else distributions)}
    result = []
    for name, dist in sorted(installed.items()):
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            dependency = installed.get(canonicalize_name(req.name))
            version = dependency.version if dependency is not None else None
            if version is None or (req.specifier and not req.specifier.contains(version, prereleases=True)):
                req.marker = None
                result.append({'package': name, 'requirement': str(req), 'installed': version})
    return result


def unknown(found: list[dict], allowed: list[dict]) -> list[dict]:
    def key(row):
        return row['package'], str(Requirement(row['requirement'])), row.get('installed')
    known = {key(row) for row in allowed}
    return [row for row in found if key(row) not in known]


if __name__ == '__main__':
    rows = unknown(installed_conflicts(), json.loads(sys.argv[1]))
    if rows:
        sys.exit('unknown dependency metadata conflicts: ' + json.dumps(rows, sort_keys=True))
