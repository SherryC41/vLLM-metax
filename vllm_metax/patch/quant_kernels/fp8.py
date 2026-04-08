# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# import torch
# from enum import Enum

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

import vllm.model_executor.layers.fused_moe.oracle.fp8 as vllm_fp8

from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
)


def maca_backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    kernels = backend_to_kernel_cls(backend)
    if backend == Fp8MoeBackend.TRITON:
        # ┌------------------------  Metax Modification -------------------------┐
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts as mx_TritonExperts,
        )

        kernels = [mx_TritonExperts]
        # └------------------------- Metax Modification -------------------------┘
    return kernels


vllm_fp8.backend_to_kernel_cls = maca_backend_to_kernel_cls
