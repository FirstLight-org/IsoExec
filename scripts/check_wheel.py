#!/usr/bin/env python3
"""Check the wheel produced by make build.

Install it outside the checkout so local code cannot hide a broken package. The
wheel is built from the sdist, so skip repeated checks from sdist. The bundled
CUTLASS tree is compared with the pinned commit's tree, which git fetches (trees only).
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import zipfile

SUFFIXES = {'.py', '.cu', '.cuh', '.cpp', '.h', '.hpp', '.json', '.yaml', '.txt'}

IMPORT_CHECK = '''
import importlib.metadata, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import isoexec
assert Path(isoexec.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), isoexec.__file__
assert isoexec.__version__ == importlib.metadata.version('isoexec')
assert 'torch' not in sys.modules
# isoexec-setup loads the bundled installer and finds its manifest without a runtime dependency.
entry, = importlib.metadata.distribution('isoexec').entry_points.select(
    group='console_scripts', name='isoexec-setup')
installer = sys.modules[entry.load().__module__]
assert installer.__name__ == 'isoexec._runtime.setup_env' and installer.INSTALLED, installer.__file__
assert installer.MANIFEST.is_file(), installer.MANIFEST
assert 'torch' not in sys.modules
# The bundled CUTLASS headers land where isoexec.ops.build_cache.CUTLASS_BUNDLED looks for them.
assert (Path(isoexec.__file__).parent / '_runtime/cutlass/include/cutlass/version.h').is_file()
'''

PUBLIC_API_CHECK = '''
import importlib.abc
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'skyrl', 'vllm', 'megatron'}:
            raise AssertionError('framework imported by public API: ' + fullname)
sys.meta_path.insert(0, Boundary())
import isoexec
from isoexec import IsoExecConfig, ModelSource, ModelSourceError
assert set(isoexec.__all__) == {'IsoExecConfig', 'ModelSource', 'ModelSourceError'}, isoexec.__all__
assert issubclass(ModelSourceError, ValueError)
assert IsoExecConfig().logprob_bound == 0.0
import torch
from isoexec.models.qwen3_5_moe.tests.tiny import TINY_MOE_HF
source = ModelSource.from_dict({'package': 'isoexec.models.qwen3_5_moe'})
model = source.build_meta(TINY_MOE_HF)
assert all(parameter.is_meta for parameter in model.parameters())
from isoexec.core.bindings import BoundOp
from isoexec.ops.table import OpTable
binding = OpTable(mode='reference').bind('norms.rms.pre_attn', site='smoke.norm',
                                        convention='rms', inputs={'eps': 1e-6})
assert isinstance(binding, BoundOp) and callable(binding.fn)
assert not torch.cuda.is_initialized()
'''


#: Files setup.py writes into the wheel's isoexec/_runtime, and their sources; both are required.
BUNDLED = {'isoexec/_runtime/setup_env.py': 'scripts/setup_env.py',
           'isoexec/_runtime/default.toml': 'environments/default.toml'}


#: The CUTLASS tree setup.py checks out into the wheel; it is not in the source, so it is checked
#: against the source manifest's [cutlass] pin instead.
CUTLASS = 'isoexec/_runtime/cutlass/'


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


@cache
def cutlass_blobs(repository: str, revision: str) -> dict[str, str]:
    """Path -> git blob ID of the pinned commit's include/, tools/util/include/ and LICENSE.txt."""
    with tempfile.TemporaryDirectory(prefix='isoexec-cutlass-check-') as temporary:
        def git(*args: str) -> bytes:
            return subprocess.run(['git', '-C', temporary, *args], check=True, stdout=subprocess.PIPE).stdout

        git('init', '-q')
        git('fetch', '-q', '--depth', '1', '--filter=blob:none', repository, revision)
        found = git('rev-parse', 'FETCH_HEAD').decode().strip()
        if found != revision:
            raise ValueError(f'fetched CUTLASS is {found}, not {revision}')
        entries = git('ls-tree', '-r', '-z', '--full-tree', revision, '--',
                      'include', 'tools/util/include', 'LICENSE.txt')
    blobs = {}
    for entry in filter(None, entries.split(b'\0')):
        metadata, path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        if kind != b'blob' or mode not in (b'100644', b'100755'):
            raise ValueError(f'unsupported CUTLASS tree entry: {entry!r}')
        blobs[os.fsdecode(path)] = oid.decode()
    return blobs


def check_cutlass(archive: zipfile.ZipFile, source: Path) -> int:
    """The bundled CUTLASS tree is exactly the pinned commit's headers and license, byte for byte."""
    spec = tomllib.loads((source / 'environments/default.toml').read_text())['cutlass']
    names = {name[len(CUTLASS):] for name in archive.namelist()
             if name.startswith(CUTLASS) and not name.endswith('/')}
    if '.isoexec-revision' not in names:
        raise ValueError(f'wheel lacks {CUTLASS}.isoexec-revision')
    found = archive.read(CUTLASS + '.isoexec-revision').decode().strip()
    if found != spec['revision']:
        raise ValueError(f"bundled CUTLASS is {found}, not the manifest pin {spec['revision']}")
    expected = cutlass_blobs(spec['repository'], spec['revision'])
    missing = sorted(set(expected) - names)
    if missing:
        raise ValueError(f'wheel lacks {[CUTLASS + name for name in missing[:5]]} ({len(missing)} files)')
    extra = sorted(names - set(expected) - {'.isoexec-revision'})
    if extra:
        raise ValueError(f'unexpected files in {CUTLASS}: {extra[:5]}')
    stale = [name for name, oid in sorted(expected.items()) if git_blob_id(archive.read(CUTLASS + name)) != oid]
    if stale:
        raise ValueError(f"bundled CUTLASS bytes differ from {spec['revision']}: {stale[:5]}")
    return len(names)


def compare_with_source(wheel: Path, source: Path) -> int:
    expected = {p.relative_to(source).as_posix(): p.read_bytes() for p in (source / 'isoexec').rglob('*')
                if p.is_file() and p.suffix in SUFFIXES}
    expected |= {name: (source / origin).read_bytes() for name, origin in BUNDLED.items()}
    with zipfile.ZipFile(wheel) as archive:
        check_cutlass(archive, source)
        actual = {name: archive.read(name) for name in archive.namelist()
                  if name.startswith('isoexec/') and not name.startswith(CUTLASS)
                  and (Path(name).suffix in SUFFIXES or name in BUNDLED)}
    if not expected or expected.keys() != actual.keys():
        raise ValueError(f'wheel/source file lists differ: {sorted(expected.keys() ^ actual.keys())}')
    stale = sorted(name for name in expected if expected[name] != actual[name])
    if stale:
        raise ValueError(f'wheel contains stale bytes: {stale}')
    return len(expected)


def install_and_import(wheel: Path, *, public_api: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix='isoexec-installed-') as target:
        subprocess.run(['uv', 'pip', 'install', '--no-config', '--python', sys.executable, '--target', target,
                        '--no-deps', str(wheel)], check=True, capture_output=True)
        # -S hides site-packages, so importing isoexec must not need a runtime dependency.
        # The public API needs the caller's Torch, so it keeps site-packages.
        flags = ['-I'] if public_api else ['-I', '-S']
        script = IMPORT_CHECK + (PUBLIC_API_CHECK if public_api else '')
        result = subprocess.run([sys.executable, *flags, '-c', script, target], cwd=target,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'installed-wheel import failed:\n{result.stdout}{result.stderr}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--source', type=Path, help='also compare package files with this saved source')
    parser.add_argument('--public-api', action='store_true', help='test the public API; requires Torch')
    args = parser.parse_args()
    if args.source is not None:
        print(f'{compare_with_source(args.wheel, args.source)} package files match {args.source}')
        with zipfile.ZipFile(args.wheel) as archive:
            print(f'{check_cutlass(archive, args.source)} bundled CUTLASS files at the manifest pin')
    install_and_import(args.wheel, public_api=args.public_api)
    print(f'{args.wheel.name}: installed import{" and public API" if args.public_api else ""} passed')


if __name__ == '__main__':
    main()
