# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from .quant_config import DeepseekV4FP8Config

from .metax.dspark import DSparkDeepseekV4ForCausalLM
from .metax.vl_model import DeepseekV41ForCausalLM

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
