"""The managed-runtime installer behind the isoexec-setup command.

A built wheel carries scripts/setup_env.py here as setup_env.py, with environments/default.toml
beside it, and the pinned CUTLASS headers in cutlass/ (see setup.py). A source checkout or editable
install runs the checkout's script instead and has no cutlass/.
"""
