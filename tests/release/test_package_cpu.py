import base64
import csv
import hashlib
import io
import zipfile

import pytest

import check_wheel
from check_wheel import compare_with_source, install_and_import


def write_wheel(tmp_path, init=b'__version__ = "0.1.0"\n'):
    files = {
        'isoexec/__init__.py': init,
        'isoexec/kernels/example.cu': b'// kernel\n',
        'isoexec-0.1.0.dist-info/METADATA': b'Metadata-Version: 2.4\nName: isoexec\nVersion: 0.1.0\n',
        'isoexec-0.1.0.dist-info/WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    source = tmp_path / 'source'
    for name in ('isoexec/__init__.py', 'isoexec/kernels/example.cu'):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(files[name])
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
    assert compare_with_source(wheel, source) == 2
    install_and_import(wheel)


@pytest.mark.parametrize('change,error', [('stale', 'stale bytes'), ('extra', 'file lists differ')])
def test_source_mismatch_refuses(tmp_path, change, error):
    wheel, source = write_wheel(tmp_path)
    path = source / ('isoexec/kernels/example.cu' if change == 'stale' else 'isoexec/kernels/other.cu')
    path.write_text('// not in the wheel\n')
    with pytest.raises(ValueError, match=error):
        compare_with_source(wheel, source)


def test_import_cannot_use_runtime_dependencies(tmp_path, monkeypatch):
    # pytest is installed only in this environment's site-packages, like Torch.
    wheel, _ = write_wheel(tmp_path, init=b'import pytest\n__version__ = "0.1.0"\n')
    with pytest.raises(RuntimeError, match="No module named 'pytest'"):
        install_and_import(wheel)
    monkeypatch.setattr(check_wheel, 'PUBLIC_API_CHECK', '')
    install_and_import(wheel, public_api=True)
