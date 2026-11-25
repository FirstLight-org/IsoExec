"""Setuptools hook: bundle the managed-runtime installer into built wheels.

All metadata lives in pyproject.toml. build_py replaces the source-tree stand-in
isoexec/_runtime/setup_env.py with scripts/setup_env.py and adds environments/default.toml beside
it, so an installed wheel provides the isoexec-setup command. An editable install keeps the
stand-in, which runs the checkout's scripts/setup_env.py.
"""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
#: Source file -> path inside the isoexec package of a built wheel.
BUNDLED = {
    'scripts/setup_env.py': '_runtime/setup_env.py',
    'environments/default.toml': '_runtime/default.toml',
}


class BuildPyWithRuntime(build_py):
    def run(self):
        super().run()
        if self.editable_mode or self.dry_run:
            return
        package = Path(self.build_lib) / 'isoexec'
        for source, target in BUNDLED.items():
            # Unconditional: copy_file skips a target that is not older than its source, and a git
            # archive gives the stand-in and scripts/setup_env.py the same mtime.
            shutil.copyfile(ROOT / source, package / target)


setup(cmdclass={'build_py': BuildPyWithRuntime})
