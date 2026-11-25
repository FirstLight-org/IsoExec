#!/usr/bin/env python3
"""Install the published IsoExec wheel, or editable IsoExec for development, in a managed root.

Run as `make setup` from a checkout, or as `isoexec-setup` after `pip install isoexec`.

The root's venv holds the pinned SkyRL lock, the pinned CUTLASS headers
and the overrides in environments/default.toml. Rerunning setup, or --check-only, checks an existing root instead of
installing again. --wheel selects a local candidate instead of the published wheel. A nonempty
root without a receipt is refused, never deleted.
"""
from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
#: Installed as isoexec/_runtime/setup_env.py (setup.py bundles it with its manifest), or run from a checkout.
INSTALLED = (HERE / 'default.toml').is_file()
REPO = None if INSTALLED else HERE.parent
MANIFEST = HERE / 'default.toml' if INSTALLED else REPO / 'environments/default.toml'
PYPI_API = 'https://pypi.org/pypi/isoexec'

# Runs in the managed venv: the runtime's own dependency check, then, with a wheel, the installed
# package must be exactly the wheel's files.
PROBE = r'''
import importlib.metadata as metadata, platform, sys, zipfile
from pathlib import Path
import torch, megatron.core, vllm, skyrl, isoexec
from isoexec.checks import dependencies
dependencies.check(host='setup')
if sys.argv[1]:
    dist = metadata.distribution('isoexec')
    with zipfile.ZipFile(sys.argv[1]) as archive:
        names = {name for name in archive.namelist() if not name.endswith('/RECORD')}
        changed = sorted(name for name in names if not Path(dist.locate_file(name)).is_file()
                         or Path(dist.locate_file(name)).read_bytes() != archive.read(name))
    package = Path(isoexec.__file__).parent
    extra = sorted({p.relative_to(package.parent).as_posix() for p in package.rglob('*')
                    if p.is_file() and p.suffix != '.pyc'} - names)
    if changed or extra:
        raise RuntimeError(f'installed isoexec differs from the wheel: changed {changed}, extra {extra}')
print(platform.python_version())
'''


class SetupError(Exception):
    """A check failed; no receipt is written."""


def log(message: str) -> None:
    print(f'[setup] {message}', flush=True)


def run(command, *, env=None, cwd=None, capture=False) -> str:
    command = [str(part) for part in command]
    log('$ ' + shlex.join(command[:3] + ['<probe>'] if '-c' in command else command))
    result = subprocess.run(command, env=env, cwd=cwd, text=True, capture_output=capture)
    if result.returncode:
        tail = (result.stdout + result.stderr)[-4000:] if capture else ''
        raise SetupError(f'{shlex.join(command[:3])} exited {result.returncode}\n{tail}')
    return result.stdout or ''


def pip(action: str, python: Path, *args) -> list:
    return ['uv', 'pip', action, '--no-config', '--python', python, *args]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest_digest(manifest: dict) -> str:
    """Hash of the parsed manifest, so comment edits do not invalidate a root."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split('.'))


def cuda_home(manifest: dict) -> str:
    """The toolkit IsoExec's own CUDA sources (vendored FA3 included) compile with.

    $CUDA_HOME, else $CUDA_PATH, else the toolkit whose nvcc is on PATH; the same order
    torch.utils.cpp_extension uses, so setup and runtime builds agree on one compiler.
    """
    configured = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if configured:
        return configured
    nvcc = shutil.which('nvcc')
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])
    raise SetupError(f"no CUDA {manifest['host']['cuda_major']} toolkit found: set CUDA_HOME, or put its "
                     "bin/ (with nvcc) on PATH")


def check_host(manifest: dict) -> dict:
    """Refuse a host that cannot build the runtime or that shows an unsupported GPU.

    Building needs no GPU, so a host without one, or with an old driver, only gets a warning.
    """
    host = manifest['host']
    facts = {'platform': f'{platform.system().lower()}-{platform.machine()}'}
    if facts['platform'] != host['platform']:
        raise SetupError(f"incompatible host: platform {facts['platform']} != {host['platform']}")
    libc, libc_version = platform.libc_ver()
    facts['glibc'] = libc_version if libc == 'glibc' else 'unavailable'
    missing = [name for name in ('git', 'gcc', 'g++') if shutil.which(name) is None]
    if missing:
        raise SetupError(f"missing host commands: {', '.join(missing)}")
    facts['cuda_home'] = cuda_home(manifest)
    nvcc = run([Path(facts['cuda_home']) / 'bin/nvcc', '--version'], capture=True)
    facts['nvcc'] = re.search(r'V(\d+\.\d+\.\d+)', nvcc).group(1)
    try:
        rows = run(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'],
                   capture=True) if shutil.which('nvidia-smi') else ''
    except SetupError:
        rows = ''
    facts['gpus'] = [[field.strip() for field in row.split(',')] for row in rows.splitlines() if row.strip()]
    problems = []
    if version_tuple(facts['nvcc'])[0] != host['cuda_major']:
        problems.append(f"nvcc {facts['nvcc']} is not CUDA {host['cuda_major']}")
    # Temporary: SkyRL's _set_numa_affinity calls CDLL(find_library('numa')). Without libnuma that is
    # CDLL(None), which loads the main program instead of raising, so the next numa_* lookup kills the
    # trainer worker. Remove once the pinned SkyRL revision skips NUMA affinity without libnuma.
    if ctypes.util.find_library('numa') is None:
        problems.append('libnuma not found (Debian/Ubuntu: libnuma1)')
    # isoexec/core/arch.py tags every major-10 GPU as sm100, so the runtime would not refuse a B300 (10.3).
    for name, driver, capability in facts['gpus']:
        if capability not in host['compute_capabilities']:
            problems.append(f"{name} compute capability {capability} is not one of "
                            f"{', '.join(host['compute_capabilities'])}")
    if problems:
        raise SetupError('incompatible host: ' + '; '.join(problems))
    if not facts['gpus']:
        log('warning: no visible NVIDIA GPU; the runtime needs one, setup does not')
    for name, driver, _ in facts['gpus']:
        if version_tuple(driver) < version_tuple(host['driver_minimum']):
            log(f"warning: {name} driver {driver} < {host['driver_minimum']}; CUDA 13 needs a newer driver")
    return facts


def project_version() -> str:
    if INSTALLED:
        from importlib import metadata
        return metadata.version('isoexec')
    return tomllib.loads((REPO / 'pyproject.toml').read_text())['project']['version']


def check_wheel_version(path: Path, version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata_files) != 1:
            raise SetupError('wheel must contain exactly one METADATA file')
        fields = dict(line.split(': ', 1) for line in archive.read(metadata_files[0]).decode().splitlines()
                      if ': ' in line)
    if fields.get('Name') != 'isoexec' or fields.get('Version') != version:
        raise SetupError(f"expected isoexec {version}, got {fields.get('Name')} {fields.get('Version')}")


def download_release_wheel(directory: Path) -> Path:
    """Download this version's wheel from PyPI and verify PyPI's SHA256 digest."""
    version = project_version()
    request = urllib.request.Request(f'{PYPI_API}/{version}/json', headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            release = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SetupError(f'isoexec {version} is not on PyPI; pass WHEEL=/path/to/candidate.whl') from error
        raise SetupError(f'PyPI lookup failed: HTTP {error.code}') from error
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise SetupError(f'PyPI lookup failed: {error}') from error

    files = [item for item in release.get('urls', []) if item.get('packagetype') == 'bdist_wheel']
    if len(files) != 1:
        raise SetupError(f'isoexec {version} must have exactly one wheel on PyPI, found {len(files)}')
    item = files[0]
    digest = item.get('digests', {}).get('sha256', '')
    if re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise SetupError(f"PyPI did not provide a SHA256 digest for {item.get('filename')}")

    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f"{item['filename']}.partial"
    path = directory / item['filename']
    try:
        with urllib.request.urlopen(item['url'], timeout=60) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        raise SetupError(f"failed to download {item['filename']}: {error}") from error
    if sha256(partial) != digest:
        partial.unlink()
        raise SetupError(f"downloaded {item['filename']} does not match PyPI's SHA256 digest")
    partial.replace(path)
    check_wheel_version(path, version)
    return path


def skyrl_checkout(manifest: dict, path: Path, env: dict, *, clone: bool) -> Path:
    """The clean SkyRL checkout at the pinned revision and lock."""
    spec = manifest['skyrl']
    if clone and not (path / '.git').is_dir():
        path.mkdir(parents=True, exist_ok=True)
        run(['git', 'init', '-q', path], env=env)
        run(['git', '-C', path, 'fetch', '-q', '--depth', '1', spec['repository'], spec['revision']], env=env)
        run(['git', '-C', path, 'checkout', '-q', 'FETCH_HEAD'], env=env)
    if run(['git', '-C', path, 'rev-parse', 'HEAD'], env=env, capture=True).strip() != spec['revision']:
        raise SetupError(f"SkyRL checkout {path} is not at {spec['revision']}")
    if run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'], env=env, capture=True).strip():
        raise SetupError(f'SkyRL checkout {path} has local modifications')
    if sha256(path / 'uv.lock') != spec['lock_sha256']:
        raise SetupError(f'SkyRL uv.lock in {path} differs from lock_sha256')
    return path


CUTLASS_PARTS = ('include', 'tools/util/include')
CUTLASS_REVISION_FILE = '.isoexec-revision'  # isoexec.ops.build_cache.CUTLASS_REVISION_FILE


def cutlass_headers(manifest: dict, venv: Path, env: dict, *, fetch: bool) -> Path:
    """CUTLASS headers at the pinned revision, where isoexec.ops.build_cache finds them in the venv."""
    spec = manifest['cutlass']
    path = venv / 'share/isoexec/cutlass'
    if fetch:
        path.mkdir(parents=True)
        run(['git', 'init', '-q', path], env=env)
        run(['git', '-C', path, 'sparse-checkout', 'set', *CUTLASS_PARTS], env=env)
        run(['git', '-C', path, 'fetch', '-q', '--depth', '1', '--filter=blob:none', spec['repository'],
             spec['revision']], env=env)
        run(['git', '-C', path, 'checkout', '-q', 'FETCH_HEAD'], env=env)
        (path / CUTLASS_REVISION_FILE).write_text(spec['revision'] + '\n')
    if not (path / 'include/cutlass/version.h').is_file():
        raise SetupError(f'CUTLASS headers are missing from {path}; choose a new ISOEXEC_ROOT')
    if run(['git', '-C', path, 'rev-parse', 'HEAD'], env=env, capture=True).strip() != spec['revision']:
        raise SetupError(f"CUTLASS checkout {path} is not at {spec['revision']}")
    if run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'], env=env, capture=True).strip():
        raise SetupError(f'CUTLASS checkout {path} has local modifications')
    if (path / CUTLASS_REVISION_FILE).read_text().strip() != spec['revision']:
        raise SetupError(f'{path / CUTLASS_REVISION_FILE} does not record {spec["revision"]}')
    return path


def process_env(manifest: dict, root: Path, venv: Path) -> dict:
    env = dict(os.environ)
    for name in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'UV_PROJECT', 'UV_WORKING_DIRECTORY'):
        env.pop(name, None)
    for name, relative in manifest['caches'].items():
        env[name] = str(root / relative)
    env.update(manifest['env'], UV_PROJECT_ENVIRONMENT=str(venv), ISOEXEC_ROOT=str(root),
               CUDA_HOME=cuda_home(manifest))
    env['PATH'] = str(venv / 'bin') + os.pathsep + env.get('PATH', '')
    return env


def install(manifest: dict, skyrl: Path, root: Path, env: dict, wheel: Path | None) -> None:
    """Without a wheel, install editable IsoExec and the dev tools."""
    python = root / 'venv/bin/python'
    # uv downloads this exact Python when the host does not have it.
    run(['uv', 'venv', '--no-config', '-q', '--python', manifest['host']['python'], root / 'venv'], env=env)
    cutlass_headers(manifest, root / 'venv', env, fetch=True)
    extras = [flag for extra in manifest['skyrl']['extras'] for flag in ('--extra', extra)]
    run(['uv', 'sync', '--project', skyrl, '--frozen', '--inexact', *extras], env=env, cwd=skyrl)
    run(pip('install', python, '--no-deps', *(['-e', REPO] if wheel is None else [wheel])), env=env)
    install_overrides(manifest, root, python, env)
    if wheel is None:
        project = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']
        run(pip('install', python, *project['optional-dependencies']['dev']), env=env)


#: Run in the managed venv: every file the named distribution recorded still has its recorded hash.
INTACT_PROBE = r"""
import base64, hashlib, sys
from importlib import metadata
dist = metadata.distribution(sys.argv[1])
stale = []
for f in dist.files or ():
    if f.hash is None or f.hash.mode != 'sha256':
        continue
    data = dist.locate_file(f).read_bytes()
    if base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode() != f.hash.value:
        stale.append(str(f))
print(len(stale), *stale[:5])
"""


def install_overrides(manifest: dict, root: Path, python: Path, env: dict) -> None:
    """Replace the lock's packages with [overrides]: uninstall, ``install``, then ``install_last``.

    ``install_last`` is a separate, later invocation so its files deterministically overwrite any path
    an earlier wheel also wrote; ``verify_intact`` names the distribution that must then own every file
    it recorded.
    """
    overrides = manifest['overrides']
    run(pip('uninstall', python, *overrides['uninstall']), env=env)
    for name, key in (('overrides.txt', 'install'), ('overrides-last.txt', 'install_last')):
        lines = overrides.get(key, [])
        if lines:
            (root / name).write_text(''.join(line + '\n' for line in lines))
            run(pip('install', python, '--no-deps', '--require-hashes', '-r', root / name), env=env)
    intact = overrides.get('verify_intact')
    if intact:
        stale = run([python, '-I', '-c', INTACT_PROBE, intact], env=env, capture=True).splitlines()[-1]
        if not stale.startswith('0'):
            raise SetupError(f'{intact}: files overwritten by another wheel after the overrides: {stale}')


def verify(manifest: dict, root: Path, env: dict, wheel: Path | None) -> list[str]:
    """Run isoexec.checks.dependencies.check() in the managed venv and return its package list.

    That check, not this script, holds the accepted versions; it reads the libnccl torch loads.
    """
    python = root / 'venv/bin/python'
    cutlass_headers(manifest, root / 'venv', env, fetch=False)
    found = run([python, '-I', '-c', PROBE, wheel or ''], env=env, cwd=root, capture=True).splitlines()[-1]
    if found != manifest['host']['python']:
        raise SetupError(f"venv Python {found} != {manifest['host']['python']}")
    return run(pip('freeze', python), env=env, capture=True).splitlines()


def check_metadata_conflicts(manifest: dict, root: Path, env: dict) -> None:
    """Check [[metadata_conflicts]] against the installed lock.

    The conflicts are fixed by this commit and lock_sha256, so only make setup-dev runs this check,
    for maintainers who change the lock or [overrides]; user setup does not rerun it.
    """
    run([root / 'venv/bin/python', '-I', REPO / 'scripts/check_runtime_dependency_conflicts.py',
         json.dumps(manifest['metadata_conflicts'])], env=env, cwd=root)


def setup(args, manifest: dict) -> None:
    root = args.root.expanduser().resolve()
    env = process_env(manifest, root, root / 'venv')
    receipt = root / 'receipt.json'
    wheel = args.wheel.expanduser().resolve() if args.wheel else None
    if receipt.is_file():
        built = json.loads(receipt.read_text())
        if built['manifest_digest'] != manifest_digest(manifest):
            raise SetupError('environments/default.toml changed since setup; choose a new ISOEXEC_ROOT')
        if not args.check_only:
            different = built['dev'] != args.dev
            different |= wheel is not None and built['wheel_sha256'] != sha256(wheel)
            if args.skyrl_source:
                different |= built['skyrl'] != str(args.skyrl_source.expanduser().resolve())
            if different:
                raise SetupError(f'{root} holds a different install; choose a new ISOEXEC_ROOT')
        try:
            check_host(manifest)
            skyrl_checkout(manifest, Path(built['skyrl']), env, clone=False)
            packages = verify(manifest, root, env, built['wheel'] and root / built['wheel'])
            if packages != built['packages']:
                raise SetupError(f"packages changed since setup: {sorted(set(packages) ^ set(built['packages']))}")
        except Exception:
            receipt.unlink()  # A failed check must not leave an ENV_PASS behind.
            raise
        log(f'ENV_PASS: reused {root}; numerical qualification not run')
        return
    if args.check_only:
        raise SetupError(f'{receipt} is missing; run make setup (or isoexec-setup) first')
    if args.dev and INSTALLED:
        raise SetupError('--dev needs a source checkout: clone the repository and run make setup-dev')
    if root.exists() and any(root.iterdir()):
        raise SetupError(f'{root} is nonempty without a receipt; choose a new ISOEXEC_ROOT')
    version = project_version()
    if wheel:
        if not wheel.is_file():
            raise SetupError(f'{wheel} does not exist')
        check_wheel_version(wheel, version)
    facts = check_host(manifest)
    for relative in manifest['caches'].values():
        (root / relative).mkdir(parents=True, exist_ok=True)
    if not args.dev and wheel is None:
        with tempfile.TemporaryDirectory(prefix='isoexec-release-') as temporary:
            wheel = download_release_wheel(Path(temporary))
            wheel = Path(shutil.copy(wheel, root))
    elif wheel:
        wheel = Path(shutil.copy(wheel, root))
    if args.skyrl_source:
        skyrl = skyrl_checkout(manifest, args.skyrl_source.expanduser().resolve(), env, clone=False)
    else:
        skyrl = skyrl_checkout(manifest, root / 'skyrl', env, clone=True)
    install(manifest, skyrl, root, env, wheel)
    packages = verify(manifest, root, env, wheel)
    if args.dev:
        check_metadata_conflicts(manifest, root, env)
    facts['python'] = manifest['host']['python']
    receipt.write_text(json.dumps({
        'status': 'ENV_PASS', 'manifest_digest': manifest_digest(manifest), 'dev': args.dev,
        'wheel': wheel and wheel.name, 'wheel_sha256': wheel and sha256(wheel), 'skyrl': str(skyrl),
        'host': facts, 'packages': packages,
    }, indent=2) + '\n')
    log(f'ENV_PASS: {root}; numerical qualification not run')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ISOEXEC_ROOT', '~/.isoexec')),
                        help='managed runtime root (default: $ISOEXEC_ROOT or ~/.isoexec)')
    parser.add_argument('--skyrl-source', type=Path, help='existing SkyRL checkout instead of cloning into the root')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--wheel', type=Path, help='IsoExec wheel to install')
    mode.add_argument('--dev', action='store_true', help='install editable IsoExec and the dev tools')
    mode.add_argument('--check-only', action='store_true', help='check the existing root without installing')
    args = parser.parse_args(argv)
    try:
        with MANIFEST.open('rb') as stream:
            setup(args, tomllib.load(stream))
    except (SetupError, OSError) as error:
        print(f'[setup] FAILED: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
