# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.quantization import QuantizationConfig


class MacaGateLinear(GateLinear):
    """MoE gate linear layer with three-tier GEMM dispatch:

    1. MACA fp32 router kernel (batch<=32, supported dims)
    2. MACA bf16×bf16→fp32 (bf16 weight + fp32 out_dtype)
    3. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        skip_bias_add: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        return_bias: bool = True,
    ):
        # NOTE(metax): Keep the full upstream lifecycle, including quantized
        # gates, output/bias settings and state consumed by inherited methods.
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=params_dtype,
            force_fp32_compute=force_fp32_compute,
            skip_bias_add=skip_bias_add,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
        )

        # NOTE(metax): MACA supplies the fp32 router kernel and torch.mm
        # out_dtype epilogue. CUDA CuteDSL and ROCm tiers stay disabled by
        # the upstream OOT platform checks, including in set_out_dtype().
        self.allow_fp32_router_gemm = (
            self.is_unquantized
            and not bias
            and self.weight.dtype == torch.float32
            and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES
        )
        self._router_gemm_cublas_capable = self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
