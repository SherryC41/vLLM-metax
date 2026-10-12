# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: Provide a pyNCCL-compatible MCCL wrapper for MetaX distributed
#       communication.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: Upstream distributed communication supports MCCL library symbols without
#     replacing the NCCL bindings.
# -----------------------------------------------------------------------------


from vllm_metax.patch.utils import patch
from vllm_metax.distributed.device_communicators.pynccl_wrapper import MCCLLibrary

patch("vllm.distributed.device_communicators.pynccl_wrapper", "NCCLLibrary")(
    MCCLLibrary
)
