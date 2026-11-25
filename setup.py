"""Setuptools hook: bundle the managed-runtime installer into the package.

All metadata lives in pyproject.toml. This file only copies scripts/setup_env.py and
environments/default.toml into isoexec/_runtime/ of the build, so `pip install isoexec` provides the
`isoexec-setup` command. A wheel never runs this file at install time; the pinned SkyRL lock and the
`--no-deps` [overrides] are applied by `isoexec-setup`, because package metadata cannot override
another distribution's requirements and PyPI refuses Git (direct URL) dependencies.
"""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
#: Source file -> path inside the isoexec package.
BUNDLED = {
    'scripts/setup_env.py': '_runtime/setup_env.py',
    'environments/default.toml': '_runtime/default.toml',
}


class BuildPyWithRuntime(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return  # An editable checkout runs scripts/setup_env.py through make.
        package = Path(self.build_lib) / 'isoexec'
        (package / '_runtime').mkdir(parents=True, exist_ok=True)
        (package / '_runtime/__init__.py').write_text('"""Bundled copy of scripts/setup_env.py and its manifest."""\n')
        for source, target in BUNDLED.items():
            self.copy_file(str(ROOT / source), str(package / target), preserve_mode=False)


setup(cmdclass={'build_py': BuildPyWithRuntime})
