# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.platforms import PlatformEnum

register_linear_kernel(EmulationMxfp8LinearKernel, PlatformEnum.OOT, "mxfp8")
