# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Set the MetaX serving benchmark temperature default to 0.0 for deterministic
#     sampling while preserving upstream argument registration and explicit user values.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: MetaX benchmark policy adopts the upstream temperature default instead of
#     requiring deterministic sampling by default.
# -----------------------------------------------------------------------------

"""Default MetaX serving benchmarks to deterministic sampling.

Affected: vLLM 0.30.0 (ced6857afa0e). Remove if benchmark policy adopts the
upstream temperature default (None). Keep all upstream arguments and validation.
"""

from vllm.benchmarks.serve import add_cli_args as _original_add_cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_metax.patch.utils import patch


@patch("vllm.benchmarks.serve")
def add_cli_args(parser: FlexibleArgumentParser):
    result = _original_add_cli_args(parser)
    # /-------------------- MetaX Modification --------------------\
    parser.set_defaults(temperature=0.0)
    # \-------------------- MetaX Modification --------------------/
    return result
