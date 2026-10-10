# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations


from vllm.config import get_current_vllm_config
from vllm.config.quantization import QuantSpec
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp8Dynamic,
    kMxfp8Static,
)

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm_metax.registry.quant_config.fp8 import MacaFp8Config

from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config
from vllm.models.deepseek_v41.quant_config import (
    DeepseekV4FP8Config as DeepseekV41FP8Config,
)


model_type = get_current_vllm_config().model_config.hf_config.model_type


class MacaDeepseekV4FP8Config(DeepseekV4FP8Config, MacaFp8Config):
    pass


class MacaDeepseekV41FP8Config(DeepseekV41FP8Config, MacaFp8Config):
    def get_quant_method(self, layer, prefix):
        if (
            isinstance(layer, LinearBase)
            and self.weight_block_size == [32, 32]
            and self.is_scale_e8m0
        ):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()

            if prefix.rsplit(".", 1)[-1] != "wo_a":
                from vllm_metax.registry.quant_config.fp8 import MctlassFp8LinearMethod

                return MctlassFp8LinearMethod(self)

            from vllm.model_executor.layers.quantization.modelopt import (
                CkptCtx,
                ModelOptLinearMethod,
            )

            rows, cols = self.weight_block_size
            return ModelOptLinearMethod(
                QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
                CkptCtx(scale_block_size=(rows, cols)),
            )
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fall through to Fp8Config which
            # returns Fp8MoEMethod with block-wise float32 scales.
        return super().get_quant_method(layer, prefix)


if model_type == "deepseek_v4":
    register_quantization_config("deepseek_v4", MacaDeepseekV4FP8Config)
elif model_type == "deepseek_v41":
    register_quantization_config("deepseek_v4", MacaDeepseekV41FP8Config)
