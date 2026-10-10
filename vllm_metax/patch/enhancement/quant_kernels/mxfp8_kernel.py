# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Add plugin-aware scaled-MM kernel dispatch for MetaX.
#
# Affected versions: vLLM 0.30.0
#
# Remove at: Upstream register_linear_kernel supports EmulationMxfp8LinearKernel,
#            allowing this entry to move to the normal MetaX kernel registry.
# -----------------------------------------------------------------------------

from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.platforms import PlatformEnum


import vllm.model_executor.kernels.linear

vllm.model_executor.kernels.linear._POSSIBLE_MXFP8_KERNELS.setdefault(
    PlatformEnum.OOT, []
).append(EmulationMxfp8LinearKernel)
