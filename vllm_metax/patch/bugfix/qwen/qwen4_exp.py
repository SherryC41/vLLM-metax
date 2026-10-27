# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Append startup compilation for loaded MetaX QSA layers because
#     upstream QSA warmup discovers only vllm.models.qwen4_exp.nvidia modules.
#     Transformers owns Qwen4 configs and normalizes the legacy layer alias;
#     the retired local config module must not be imported during startup.
#
# Affected versions: vLLM 0.32.0 (5df63a7203cd), MetaX QSA models.
#
# Remove at: Remove the warmup wrapper when upstream startup warmup reaches
#     the MetaX QSA kernels through a supported registration hook.
# -----------------------------------------------------------------------------
"""MetaX QSA startup compilation with Transformers-owned Qwen4 configs."""

import sys
from vllm_metax.patch.utils import PATCH_MARKER, patch

_WARMUP_TARGET = "vllm.model_executor.warmup.kernel_warmup.kernel_warmup"


@patch("vllm.model_executor.warmup.kernel_warmup", "kernel_warmup")
def kernel_warmup(worker, *, process_local_only: bool = False):
    # Preserve upstream warmup and forward the process-local mode unchanged.
    original = getattr(kernel_warmup, PATCH_MARKER)[_WARMUP_TARGET]
    original(worker, process_local_only=process_local_only)
    # /-------------------- MetaX Modification --------------------\
    # Avoid importing QSA for unrelated models. The helper checks actual model
    # owners and deduplicates compilation by cache/table layout and head counts.
    # It compiles sparse attention variants; the dynamic topk selector still JITs
    # on demand. This addition also runs in process-local mode when JIT warmup is
    # enabled, matching the upstream QSA warmup gate.
    if (
        worker.vllm_config.kernel_config.enable_jit_warmup
        and "vllm_metax.models.qwen4_exp.metax.qsa" in sys.modules
    ):
        from vllm_metax.models.qwen4_exp.metax.warmup import warmup_loaded_qsa

        warmup_loaded_qsa(worker)
    # \-------------------- MetaX Modification --------------------/
