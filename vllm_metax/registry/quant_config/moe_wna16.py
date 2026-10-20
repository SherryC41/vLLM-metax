# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    is_layer_skipped_quant,
    MoeWNA16Method as vllm_MoeWNA16Method,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


# Remove configs of marlin
@register_quantization_config("moe_wna16")
class MacaMoeWNA16Config(MoeWNA16Config):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_marlin = False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm_metax.registry.quant_config.auto_awq import MacaAutoAWQConfig
            from vllm_metax.registry.quant_config.auto_gptq import MacaAutoGPTQConfig

            linear_config: QuantizationConfig
            if self.linear_quant_method == "gptq":
                linear_config = MacaAutoGPTQConfig.from_config(self.full_config)
            elif self.linear_quant_method in ("awq", "awq_marlin"):
                linear_config = MacaAutoAWQConfig.from_config(self.full_config)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")

            # The delegate is rebuilt from the raw HF quantization dict, which
            # lists shard names and never fused ones. Without the mapping it
            # resolves fused layers (qkv_proj, gate_up_proj) to
            # UnquantizedLinearMethod and the checkpoint's qweight has nowhere
            # to load.
            linear_config.packed_modules_mapping = self.packed_modules_mapping
            return linear_config.get_quant_method(layer, prefix)
        elif isinstance(layer, RoutedExperts):
            return MoeWNA16Method(self, layer.moe_config)
        return None


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class MoeWNA16Method(vllm_MoeWNA16Method):
    pass
