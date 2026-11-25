"""Source-tree stand-in for the bundled installer; setup.py replaces this file in built wheels.

An editable install or a checkout runs scripts/setup_env.py, the single source of the installer.
"""
from pathlib import Path
import runpy


def main(argv=None) -> int:
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'setup_env.py'
    return runpy.run_path(str(script))['main'](argv)
