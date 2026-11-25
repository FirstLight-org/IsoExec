"""flash-linear-attention 0.5.2's chunk gated-delta-rule, vendored for the GDN backward.

The trainer's GDN backward (``gdn_fla_backward`` and ``gdn_cpr_sm90/fla_backward.py``) runs these
kernels. IsoExec owns the source instead of importing the ``fla`` package at runtime.

The layout mirrors upstream's ``fla/`` package, so ``fla.X.Y`` is ``vendored_fla_bwd.X.Y`` here.
The upstream is flash-linear-attention / fla-core 0.5.2 (tag v0.5.2, commit 9c8e42e, MIT). That is
a different upstream from the sibling ``vendored_fla`` tree, which is vLLM's port of FLA at vLLM
commit 6f573f486 and holds the forward kernels. Mixing the two would blur both provenances.

DO NOT EDIT the copied files. Each one is ``header + rewrite(upstream)`` under the rules in
PROVENANCE.json, and the rules only redirect imports. The package ``__init__`` files and
``_compat.py`` are IsoExec-authored glue and are listed under ``authored_files`` there.
"""

# Read by utils/_decorators.py (`from .. import __version__`), as upstream's fla/__init__.py provides it.
__version__ = "0.5.2"
