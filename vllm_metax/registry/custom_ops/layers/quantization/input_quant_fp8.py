# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8 as _vllm_QuantFP8,
)


@_vllm_QuantFP8.register_oot
class QuantFP8(_vllm_QuantFP8):
    # TODO(MetaX): Fix CustomOp.register_oot conflating its class lookup key
    # ("QuantFP8") with the config op name ("quant_fp8"). It overwrites
    # cls.name, so inherited enabled() misses +quant_fp8/-quant_fp8:
    # ["none", "+quant_fp8"] falls back to native; ["all", "-quant_fp8"]
    # incorrectly stays enabled. Remove this override once upstream preserves
    # the config op name separately from the OOT class key.
    @classmethod
    def enabled(cls) -> bool:
        return _vllm_QuantFP8.enabled()

    # TODO(MetaX): Investigate the v0.30.0 native grouped FP8 quantization trap
    # during DeepSeek V4 breakable CUDA Graph startup warmup on C600-U
    # (PyTorch 2.10.0+metax3.9.0.5, MACA 3.9.0.17). The device reports an ATU
    # fault in elementwise_kernel_2_2_cast with DivFunctor<float>. Synchronizing
    # small-batch warmup around QuantFP8 avoids it; routing all OOT calls to
    # forward_cuda passes complete capture and inference without extra sync.
    # Standalone native quantization tests pass; the underlying memory/stream
    # cause is unresolved. Keep this CUDA route until the native path is fixed
    # and validated through full startup capture and graph replay.
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
