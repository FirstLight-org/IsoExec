"""Setuptools hook: bundle the managed-runtime installer and the CUTLASS headers into built wheels.

All metadata lives in pyproject.toml. build_py replaces the source-tree stand-in
isoexec/_runtime/setup_env.py with scripts/setup_env.py, adds environments/default.toml beside it, and
checks out the [cutlass] commit that manifest pins (git, by SHA) into isoexec/_runtime/cutlass. An
installed wheel therefore provides the isoexec-setup command and the headers IsoExec's CUDA sources
compile against (isoexec.ops.build_cache.CUTLASS_BUNDLED), in any environment it is installed into.
Building a wheel needs git and network access. An editable install keeps the stand-in and fetches
nothing; managed setup installs the same headers into its venv instead.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
#: Source file -> path inside the isoexec package of a built wheel.
BUNDLED = {
    'scripts/setup_env.py': '_runtime/setup_env.py',
    'environments/default.toml': '_runtime/default.toml',
}
#: The CUTLASS parts IsoExec includes (scripts/setup_env.py CUTLASS_PARTS), and the revision marker that
#: isoexec.ops.build_cache checks against CUTLASS_REVISION.
CUTLASS_PARTS = ('include', 'tools/util/include')
CUTLASS_REVISION_FILE = '.isoexec-revision'


def fetch_cutlass(checkout: Path) -> str:
    """Check out the headers of the [cutlass] commit in environments/default.toml; return the commit."""
    spec = tomllib.loads((ROOT / 'environments/default.toml').read_text())['cutlass']

    def git(*args: str) -> str:
        return subprocess.run(['git', '-C', str(checkout), *args], check=True, text=True,
                              stdout=subprocess.PIPE).stdout

    checkout.mkdir(parents=True)
    git('init', '-q')
    git('sparse-checkout', 'set', *CUTLASS_PARTS)
    git('fetch', '-q', '--depth', '1', '--filter=blob:none', spec['repository'], spec['revision'])
    git('checkout', '-q', 'FETCH_HEAD')
    head = git('rev-parse', 'HEAD').strip()
    if head != spec['revision']:
        raise RuntimeError(f"CUTLASS checkout is at {head}, not the pinned {spec['revision']}")
    return head


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
        cutlass = package / '_runtime' / 'cutlass'
        shutil.rmtree(cutlass, ignore_errors=True)  # A stale tree from an earlier build in build_lib.
        with tempfile.TemporaryDirectory(prefix='isoexec-cutlass-') as temporary:
            checkout = Path(temporary) / 'cutlass'
            revision = fetch_cutlass(checkout)
            for part in CUTLASS_PARTS:
                shutil.copytree(checkout / part, cutlass / part)
            shutil.copyfile(checkout / 'LICENSE.txt', cutlass / 'LICENSE.txt')
        (cutlass / CUTLASS_REVISION_FILE).write_text(revision + '\n')


setup(cmdclass={'build_py': BuildPyWithRuntime})
