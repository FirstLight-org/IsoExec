"""Managed setup refusals and the dependency conflict check, without a GPU runtime."""
import json
from pathlib import Path
import runpy
import tomllib
from types import SimpleNamespace
import zipfile

import pytest

from check_runtime_dependency_conflicts import installed_conflicts, unknown
import setup_env


@pytest.fixture
def manifest():
    return tomllib.loads(setup_env.MANIFEST.read_text())


def write_receipt(root, manifest):
    root.mkdir()
    receipt = {'status': 'ENV_PASS', 'manifest_digest': setup_env.manifest_digest(manifest), 'dev': True,
               'wheel': None, 'wheel_sha256': None, 'skyrl': str(root / 'skyrl'), 'packages': []}
    (root / 'receipt.json').write_text(json.dumps(receipt))
    return receipt


def test_nonempty_root_without_receipt_is_kept(tmp_path):
    kept = tmp_path / 'root/venv/keep'
    kept.parent.mkdir(parents=True)
    kept.write_text('user data')
    assert setup_env.main(['--root', str(tmp_path / 'root'), '--dev']) == 1
    assert kept.read_text() == 'user data' and not (tmp_path / 'root/receipt.json').exists()


def test_rerun_with_another_install_keeps_the_receipt(tmp_path, manifest):
    receipt = write_receipt(tmp_path / 'root', manifest)
    wheel = tmp_path / 'isoexec-0.1.0-py3-none-any.whl'
    wheel.write_bytes(b'another wheel')
    assert setup_env.main(['--root', str(tmp_path / 'root'), '--wheel', str(wheel)]) == 1
    assert json.loads((tmp_path / 'root/receipt.json').read_text()) == receipt


def test_failed_check_removes_the_receipt(tmp_path, manifest):
    # This root has no managed venv, so the check fails on any host.
    write_receipt(tmp_path / 'root', manifest)
    assert setup_env.main(['--root', str(tmp_path / 'root'), '--check-only']) == 1
    assert not (tmp_path / 'root/receipt.json').exists()


def test_failed_download_leaves_the_root_retryable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CUDA_HOME', str(tmp_path / 'cuda'))
    monkeypatch.setattr(setup_env, 'check_host', lambda manifest: {})

    def offline(directory):
        raise setup_env.SetupError('isoexec 0.1.0 is not on PyPI')
    monkeypatch.setattr(setup_env, 'download_release_wheel', offline)
    assert setup_env.main(['--root', str(tmp_path / 'root')]) == 1
    assert 'not on PyPI' in capsys.readouterr().err
    assert not (tmp_path / 'root').exists()


def test_reuse_refuses_a_root_built_for_another_version(tmp_path, monkeypatch, manifest, capsys):
    monkeypatch.setenv('CUDA_HOME', str(tmp_path / 'cuda'))
    root = tmp_path / 'root'
    receipt = write_receipt(root, manifest)
    old = 'isoexec-0.0.9-py3-none-any.whl'
    with zipfile.ZipFile(root / old, 'w') as archive:
        archive.writestr('isoexec-0.0.9.dist-info/METADATA',
                         'Metadata-Version: 2.4\nName: isoexec\nVersion: 0.0.9\n\nVersion: 0.1.0 in the README body\n')
    receipt |= {'dev': False, 'wheel': old, 'wheel_sha256': setup_env.sha256(root / old)}
    (root / 'receipt.json').write_text(json.dumps(receipt))
    assert setup_env.main(['--root', str(root)]) == 1
    assert 'holds isoexec 0.0.9, not 0.1.0' in capsys.readouterr().err
    assert json.loads((root / 'receipt.json').read_text()) == receipt


def test_installed_installer_refuses_dev(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup_env, 'INSTALLED', True)
    assert setup_env.main(['--root', str(tmp_path / 'root'), '--dev']) == 1
    assert 'source checkout' in capsys.readouterr().err
    assert not (tmp_path / 'root').exists()


def test_source_stand_in_runs_the_checkout_installer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CUDA_HOME', str(tmp_path / 'cuda'))
    stand_in = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'isoexec/_runtime/setup_env.py'))
    assert stand_in['main'](['--root', str(tmp_path / 'root'), '--check-only']) == 1
    assert 'receipt.json is missing' in capsys.readouterr().err


def test_only_conflicts_listed_in_the_manifest_pass(manifest):
    def dist(name, version, *requires):
        return SimpleNamespace(metadata={'Name': name}, version=version, requires=list(requires))
    installed = [dist('skyrl', '0.1.0', 'transformers>=5.6.1,<=5.16.1'), dist('transformers', '5.22.0'),
                 dist('vllm', '0.30.0', 'torch==2.9.0', 'pytest; extra == "test"'), dist('torch', '2.13.0')]
    assert unknown(installed_conflicts(installed), manifest['metadata_conflicts']) == [
        {'package': 'vllm', 'requirement': 'torch==2.9.0', 'installed': '2.13.0'}]


def _fake_toolkit(root):
    (root / 'bin').mkdir(parents=True)
    nvcc = root / 'bin/nvcc'
    nvcc.write_text('#!/bin/sh\n')
    nvcc.chmod(0o755)
    return root


@pytest.mark.parametrize('variable', ['CUDA_HOME', 'CUDA_PATH'])
def test_cuda_home_honors_the_environment_first(tmp_path, monkeypatch, manifest, variable):
    for name in ('CUDA_HOME', 'CUDA_PATH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, str(tmp_path / 'explicit'))
    monkeypatch.setenv('PATH', str(_fake_toolkit(tmp_path / 'on-path') / 'bin'))
    assert setup_env.cuda_home(manifest) == str(tmp_path / 'explicit')


def test_cuda_home_falls_back_to_nvcc_on_path(tmp_path, monkeypatch, manifest):
    for name in ('CUDA_HOME', 'CUDA_PATH'):
        monkeypatch.delenv(name, raising=False)
    toolkit = _fake_toolkit(tmp_path / 'toolkit')
    monkeypatch.setenv('PATH', str(toolkit / 'bin'))
    assert setup_env.cuda_home(manifest) == str(toolkit.resolve())


def test_cuda_home_without_a_toolkit_is_refused(tmp_path, monkeypatch, manifest):
    for name in ('CUDA_HOME', 'CUDA_PATH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(setup_env.SetupError, match='set CUDA_HOME'):
        setup_env.cuda_home(manifest)


def test_cute_dsl_comes_from_the_lock_not_the_overrides(manifest):
    """CuTe DSL 4.8.0's libs wheels write disjoint paths, so the lock's install needs no ordering fix.

    Note (Alexander): 4.5.2's libs-base and libs-cu13 wrote 99 shared paths, which is what
    ``install_last`` / ``verify_intact`` exist for; with the lock's 4.8.0 nothing is overridden.
    """
    overrides = manifest['overrides']
    assert not any('cutlass' in line for line in overrides.get('install', []))
    assert not overrides.get('install_last') and not overrides.get('verify_intact')


def _two_stage_manifest(manifest):
    """The real manifest with a two-stage override, to test the mechanism ``install_last`` provides."""
    overrides = {
        'uninstall': ['nvidia-cutlass-dsl-libs-base', 'nvidia-cutlass-dsl-libs-cu13'],
        'install': ['nvidia-cutlass-dsl-libs-base==4.8.0 --hash=sha256:' + '0' * 64],
        'install_last': ['nvidia-cutlass-dsl-libs-cu13==4.8.0 --hash=sha256:' + '1' * 64],
        'verify_intact': 'nvidia-cutlass-dsl-libs-cu13',
    }
    return {**manifest, 'overrides': overrides}


def test_install_overrides_runs_the_two_installs_in_order_then_checks_the_cu13_files(tmp_path, monkeypatch, manifest):
    calls = []

    def run(command, *, env=None, cwd=None, capture=False):
        calls.append([str(part) for part in command])
        return '0\n' if capture else ''

    manifest = _two_stage_manifest(manifest)
    monkeypatch.setattr(setup_env, 'run', run)
    setup_env.install_overrides(manifest, tmp_path, tmp_path / 'venv/bin/python', {})
    assert [c[2] for c in calls[:3]] == ['uninstall', 'install', 'install']
    assert calls[1][-1].endswith('overrides.txt') and calls[2][-1].endswith('overrides-last.txt')
    assert 'libs-cu13' in (tmp_path / 'overrides-last.txt').read_text()
    assert 'libs-cu13' not in (tmp_path / 'overrides.txt').read_text()
    assert calls[3][-1] == 'nvidia-cutlass-dsl-libs-cu13'

    monkeypatch.setattr(setup_env, 'run', lambda command, **k: '3 a.py b.py c.py\n' if k.get('capture') else '')
    with pytest.raises(setup_env.SetupError, match='overwritten'):
        setup_env.install_overrides(manifest, tmp_path, tmp_path / 'venv/bin/python', {})
