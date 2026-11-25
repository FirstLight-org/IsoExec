"""Pin reference selection for CPU composition tests. Accelerator-resolution tests explicitly
override this fixture.
"""

from __future__ import annotations

from isoexec.ops.tests._arch_pin import _reference_composition  # noqa: F401  (autouse here)
