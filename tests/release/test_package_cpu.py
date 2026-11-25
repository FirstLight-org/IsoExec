import base64
import csv
import hashlib
import io
import zipfile

import pytest

import check_wheel
from check_wheel import compare_with_source, install_and_import


INSTALLER = b"""from pathlib import Path
INSTALLED = __package__ == 'isoexec._runtime'
MANIFEST = Path(__file__).with_name('default.toml')
def main(argv=None):
    return 0
"""


def write_wheel(tmp_path, init=b'__version__ = "0.1.0"\n'):
    files = {
        'isoexec/__init__.py': init,
        'isoexec/kernels/example.cu': b'// kernel\n',
        'isoexec/_runtime/__init__.py': b'',
        'isoexec/_runtime/setup_env.py': INSTALLER,
        'isoexec/_runtime/default.toml': b'schema = 1\n',
        'isoexec-0.1.0.dist-info/METADATA': b'Metadata-Version: 2.4\nName: isoexec\nVersion: 0.1.0\n',
        'isoexec-0.1.0.dist-info/WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        'isoexec-0.1.0.dist-info/entry_points.txt':
            b'[console_scripts]\nisoexec-setup = isoexec._runtime.setup_env:main\n',
    }
    source = tmp_path / 'source'
    # Source path -> wheel path; setup.py copies the last two over the stand-in.
    sources = {name: name for name in ('isoexec/__init__.py', 'isoexec/kernels/example.cu',
                                       'isoexec/_runtime/__init__.py')}
    sources |= {'isoexec/_runtime/setup_env.py': None, 'scripts/setup_env.py': 'isoexec/_runtime/setup_env.py',
                'environments/default.toml': 'isoexec/_runtime/default.toml'}
    for name, packed in sources.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(files[packed] if packed else b'# source-tree stand-in\n')
    record = io.StringIO()
    for name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
        csv.writer(record).writerow((name, 'sha256=' + digest, len(content)))
    csv.writer(record).writerow(('isoexec-0.1.0.dist-info/RECORD', '', ''))
    files['isoexec-0.1.0.dist-info/RECORD'] = record.getvalue().encode()
    wheel = tmp_path / 'isoexec-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return wheel, source


def test_wheel_matches_source_and_imports(tmp_path):
    wheel, source = write_wheel(tmp_path)
    assert compare_with_source(wheel, source) == 5
    install_and_import(wheel)


@pytest.mark.parametrize('change,error', [('stale', 'stale bytes'), ('extra', 'file lists differ')])
def test_source_mismatch_refuses(tmp_path, change, error):
    wheel, source = write_wheel(tmp_path)
    path = source / ('isoexec/kernels/example.cu' if change == 'stale' else 'isoexec/kernels/other.cu')
    path.write_text('// not in the wheel\n')
    with pytest.raises(ValueError, match=error):
        compare_with_source(wheel, source)


def test_missing_bundled_installer_refuses(tmp_path):
    wheel, source = write_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        kept = {name: archive.read(name) for name in archive.namelist() if name != 'isoexec/_runtime/default.toml'}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in kept.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='file lists differ'):
        compare_with_source(wheel, source)
    with pytest.raises(RuntimeError, match='default.toml'):
        install_and_import(wheel)


def test_import_cannot_use_runtime_dependencies(tmp_path, monkeypatch):
    # pytest is installed only in this environment's site-packages, like Torch.
    wheel, _ = write_wheel(tmp_path, init=b'import pytest\n__version__ = "0.1.0"\n')
    with pytest.raises(RuntimeError, match="No module named 'pytest'"):
        install_and_import(wheel)
    monkeypatch.setattr(check_wheel, 'PUBLIC_API_CHECK', '')
    install_and_import(wheel, public_api=True)
