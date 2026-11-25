"""Print ``export`` lines for the IsoExec trainer and Model Runner V2 engine environment of one model.

The launcher evaluates the output before starting Ray so every actor and
vLLM worker inherits the same flags. Values are kernel and runtime settings, never credentials.
"""

from __future__ import annotations

import argparse
import shlex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from isoexec.integrations.verl.model import checkpoint_directory
    from isoexec.runtimes.environment import resolved_environment
    from isoexec.runtimes.vllm.vllm_patches import apply_vllm_isoexec_env

    env = resolved_environment(str(checkpoint_directory(args.model)))
    apply_vllm_isoexec_env(env)
    env["VLLM_PLUGINS"] = "isoexec"
    for key in sorted(env):
        print(f"export {key}={shlex.quote(env[key])}")


if __name__ == "__main__":
    main()
