# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Grouped block-32 FP8 output projection with FP32 scales.

Requires fp8_einsum and fused inverse RoPE support for block size 32.
Weight scales are decoded once during weight post-processing. The fused
quantizer produces power-of-two activation scales stored as FP32.
"""

import torch

from vllm.utils.deep_gemm import fp8_einsum

from vllm_metax.registry.quant_config.fp8 import MctlassFp8LinearMethod

from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    deep_gemm_fp8_o_proj as vllm_fp8_o_proj,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)


def mctlass_fp8_o_proj(
    o,
    positions,
    cos_sin_cache,
    wo_a,
    wo_b,
    *,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    o_lora_rank,
):
    k = heads_per_group * (nope_dim + rope_dim)
    if n_groups < 1 or k % 32 or o_lora_rank % 32:
        raise ValueError(
            "Grouped block-32 projection requires positive groups and aligned K/D"
        )
    if wo_a.weight.shape != (n_groups * o_lora_rank, k):
        raise ValueError("Unexpected local wo_a weight shape")
    scale_rows = o_lora_rank // 32
    if wo_a.weight_scale.shape != (n_groups * scale_rows, k // 32):
        raise ValueError("Unexpected compact wo_a scale shape")

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=32,
        tma_aligned_scales=False,
    )
    if wo_a.weight_scale.dtype != torch.float32:
        raise ValueError("Block-32 einsum requires FP32 weight scales")
    weight = wo_a.weight.view(n_groups, o_lora_rank, k)
    weight_scale = wo_a.weight_scale.view(n_groups, scale_rows, k // 32)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=wo_a.quant_method.fp8_linear.config.out_dtype,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (weight, weight_scale),
        z,
        recipe=(1, 32, 32),
    )
    return wo_b(z.flatten(1))


def deep_gemm_fp8_o_proj(
    o,
    positions,
    cos_sin_cache,
    wo_a,
    wo_b,
    *,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    o_lora_rank,
    einsum_recipe,
    tma_aligned_scales,
):
    if not isinstance(wo_a.quant_method, MctlassFp8LinearMethod):
        return vllm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=einsum_recipe,
            tma_aligned_scales=tma_aligned_scales,
        )
    else:
        return mctlass_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
