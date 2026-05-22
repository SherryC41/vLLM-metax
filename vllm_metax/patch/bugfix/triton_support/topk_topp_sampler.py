# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Fall back to PyTorch top-k/top-p sampling to avoid Triton issues.
#
# Affected versions: v0.21.0
# Remove at: after `apply_top_k_top_p_triton` is fixed upstream.
# -----------------------------------------------

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)


import vllm.v1.sample.ops.topk_topp_sampler

vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p

import vllm.v1.sample.rejection_sampler

vllm.v1.sample.rejection_sampler.apply_top_k_top_p = apply_top_k_top_p

import vllm.v1.worker.gpu.sample.states

vllm.v1.worker.gpu.sample.states.apply_top_k_top_p = apply_top_k_top_p
