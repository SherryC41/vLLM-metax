# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
)

logger = init_logger(__name__)


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    else:
        raise ValueError(f"Not supported MXFP4 MoE backend: {backend.value}")


def convert_weight_to_mxfp4_moe_kernel_format(
    mxfp4_backend: Mxfp4MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    _cache_permute_indices: dict[torch.Size, torch.Tensor] | None = None,
    activation: MoEActivation | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if mxfp4_backend == Mxfp4MoeBackend.TRITON:
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    else:
        raise ValueError(
            f"Unsupported mxfp4_backend for Mxfp4MoEMethod: {mxfp4_backend}. "
            f"Expected TRTLLM, Triton, AITER, XPU, or emulation backend."
        )
