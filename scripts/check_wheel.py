#!/usr/bin/env python3
"""Check the wheel produced by make build.

Install it outside the checkout so local code cannot hide a broken package. The
wheel is built from the sdist, so skip repeated checks from sdist.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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


#: Files setup.py copies into the package from outside isoexec/ (its generated __init__.py is skipped).
BUNDLED = {'isoexec/_runtime/setup_env.py': 'scripts/setup_env.py',
           'isoexec/_runtime/default.toml': 'environments/default.toml'}


def compare_with_source(wheel: Path, source: Path) -> int:
    expected = {p.relative_to(source).as_posix(): p.read_bytes() for p in (source / 'isoexec').rglob('*')
                if p.is_file() and p.suffix in SUFFIXES}
    expected |= {name: (source / origin).read_bytes() for name, origin in BUNDLED.items()}
    with zipfile.ZipFile(wheel) as archive:
        actual = {name: archive.read(name) for name in archive.namelist()
                  if name.startswith('isoexec/') and (Path(name).suffix in SUFFIXES or name in BUNDLED)
                  and name != 'isoexec/_runtime/__init__.py'}
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
    install_and_import(args.wheel, public_api=args.public_api)
    print(f'{args.wheel.name}: installed import{" and public API" if args.public_api else ""} passed')


if __name__ == '__main__':
    main()
