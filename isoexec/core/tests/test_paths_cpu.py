"""Cache defaults follow an explicit override, then XDG_CACHE_HOME, then ~/.cache; the toolkit comes from torch."""

from __future__ import annotations

import pytest

from isoexec.core import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("XDG_CACHE_HOME", "PIK_CACHE", "ISOEXEC_TEST_CACHE"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_home_dot_cache(tmp_path):
    assert paths.user_cache_dir("a", "b.json") == tmp_path / "home" / ".cache" / "a" / "b.json"


@pytest.mark.parametrize("xdg", ["", None])
def test_unset_or_empty_xdg_falls_back_to_home(monkeypatch, tmp_path, xdg):
    if xdg is not None:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    assert paths.user_cache_dir("a") == tmp_path / "home" / ".cache" / "a"


def test_xdg_cache_home_is_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.user_cache_dir("a", "b") == tmp_path / "xdg" / "a" / "b"
    assert paths.pik_cache_dir() == tmp_path / "xdg" / "pik"


def test_env_override_names_the_whole_location(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("ISOEXEC_TEST_CACHE", str(tmp_path / "explicit"))
    assert paths.user_cache_dir("a", "b", env="ISOEXEC_TEST_CACHE") == tmp_path / "explicit"
    monkeypatch.setenv("PIK_CACHE", str(tmp_path / "pik-explicit"))
    assert paths.pik_cache_dir() == tmp_path / "pik-explicit"


def test_empty_override_uses_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ISOEXEC_TEST_CACHE", "")
    assert paths.user_cache_dir("a", env="ISOEXEC_TEST_CACHE") == tmp_path / "home" / ".cache" / "a"


def test_paths_are_not_created(tmp_path):
    paths.user_cache_dir("a")
    assert not (tmp_path / "home").exists()

