"""Import target for ``actor_rollout_ref.model.external_lib``: registers the IsoExec engine and trainer.

verl imports this module in the task runner and in every worker. Importing it opts that process
into the IsoExec Megatron engine (``VERL_ENGINE_VENDOR=isoexec``) and registers the
``isoexec_sync`` trainer mode.
"""

import os

from . import engine, trainer  # noqa: F401  (registration side effects)

os.environ["VERL_ENGINE_VENDOR"] = engine.VENDOR
