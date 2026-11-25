#!/usr/bin/env python3
"""Build committed HEAD once; never replace existing artifacts or include workspace files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(out: Path, source: Path, env: dict | None = None, *, tag: str | None = None) -> Path:
    out, source = out.resolve(), source.resolve()
    for directory in (out, source):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f'{directory} is not empty; use a new output/source directory')
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('output and source directories must be separate')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    # Use the resolved commit in every operation, even if HEAD moves during the build.
    with tempfile.TemporaryDirectory(prefix='isoexec-head-') as temporary:
        archive = Path(temporary) / 'head.tar'
        subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar', '-o', str(archive), head], check=True)
        source.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter='data')
        with (source / 'pyproject.toml').open('rb') as stream:
            version = tomllib.load(stream)['project']['version']
        if tag is not None:
            if tag != f'v{version}':
                raise ValueError(f'tag {tag!r} does not match package version {version}')
            tagged = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', f'{tag}^{{commit}}'],
                                             text=True).strip()
            if tagged != head:
                raise ValueError(f'{tag} does not identify {head}')
        inputs = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(['uv', 'build', '--no-config', '-q', str(source), '--out-dir', str(out)],
                       check=True, env=env or os.environ)
    wheels, sdists = sorted(out.glob('*.whl')), sorted(out.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError('expected exactly one sdist and one wheel')
    artifacts = {p.name: sha256(p) for p in [*sdists, *wheels]}
    receipt = {'schema': 1, 'source_commit': head, 'version': version, 'tag': tag,
               'artifacts': artifacts, 'source_files': inputs}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(artifacts.items())))
    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tag', help='require this existing tag to match HEAD and the package version')
    args = parser.parse_args()
    print(build(args.out, args.source, tag=args.tag))


if __name__ == '__main__':
    main()
