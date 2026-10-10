# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm_metax.models.deepseek_v41.quant_config import DeepseekV4FP8Config

import vllm.models.deepseek_v41

vllm.models.deepseek_v41.DeepseekV4FP8Config = DeepseekV4FP8Config
