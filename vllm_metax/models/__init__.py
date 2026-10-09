# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    )

    # NOTE(MetaX): v0.30.0 selects the generic DeepSeek MTP for V3.2 on
    # non-CUDA platforms. Keep the MetaX subclass so its sparse indexer and
    # cache implementation match the local V3.2 target model. Revisit when
    # the upstream generic decoder uses those MetaX kernels directly.
    ModelRegistry.register_model(
        "DeepseekV32MTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    )

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV2ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM", "vllm_metax.models.deepseek_v2:GlmMoeDsaForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_metax.models.deepseek_v4:DeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel", "vllm_metax.models.deepseek_v4:DeepSeekV4MTP"
    )

    ModelRegistry.register_model(
        "DSparkDraftModel", "vllm_metax.models.deepseek_v4:DSparkDeepseekV4ForCausalLM"
    )

    ModelRegistry.register_model(
        "Step3p5MTP", "vllm_metax.models.step3p5_mtp:Step3p5MTP"
    )

    ModelRegistry.register_model(
        "Step3p5ForCausalLM", "vllm_metax.models.step3p5:Step3p5ForCausalLM"
    )
