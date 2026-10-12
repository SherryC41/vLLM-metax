# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Redirect NCCL and device-management utilities to MetaX equivalents.
#     Update loaded aliases, including pynccl_wrapper.find_nccl_library, so
#     importing the utilities before plugin activation cannot bypass discovery.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: Upstream library and device-management discovery dispatches to MCCL and MX-
#     SMI on MetaX.
# -----------------------------------------------------------------------------

from vllm_metax.utils.mccl import find_mccl_library
from vllm_metax.utils import import_pymxsml
from vllm_metax.patch.utils import patch

patch("vllm.utils.nccl", "find_nccl_library")(find_mccl_library)
patch("vllm.utils.import_utils", "import_pynvml")(import_pymxsml)
