import pytest

from build_package import build


def test_existing_artifact_is_not_overwritten(tmp_path):
    """Reject the build since the artifacts already exist."""
    out = tmp_path / 'dist'
    out.mkdir()
    artifact = out / 'candidate.whl'
    artifact.write_bytes(b'keep this candidate')
    with pytest.raises(ValueError, match='not empty'):
        build(out, tmp_path / 'source')
    assert artifact.read_bytes() == b'keep this candidate'


def test_output_and_source_must_be_separate(tmp_path):
    """Reject the build since source contains outputs."""
    with pytest.raises(ValueError, match='separate'):
        build(tmp_path / 'source/dist', tmp_path / 'source')
    assert not (tmp_path / 'source').exists()
