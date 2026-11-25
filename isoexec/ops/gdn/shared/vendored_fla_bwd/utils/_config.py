# ---------------------------------------------------------------------------------------------
# VENDORED KERNEL -- DO NOT EDIT BY HAND.
# upstream_module: fla.utils._config
# upstream_dist:   flash-linear-attention / fla-core 0.5.2  (tag v0.5.2, commit 9c8e42e762fce087c27b673af4922795d9edb85e)
# upstream_url:    https://github.com/fla-org/flash-linear-attention (MIT; see NOTICE)
# provenance:      ops/gdn/shared/vendored_fla_bwd/PROVENANCE.json
# local changes:   only the rewrite_rules recorded in PROVENANCE.json (import redirects into this
#                  tree); vendored_sha256 pins this file.
# why vendored:    the trainer's GDN backward runs these kernels; IsoExec owns the source it runs
#                  instead of importing the fla package at runtime.
# ---------------------------------------------------------------------------------------------
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
try:
    FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))
except ValueError:
    FLA_TENSOR_CACHE_SIZE = 4
