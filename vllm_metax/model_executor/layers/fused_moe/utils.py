# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from typing import Any


def initialize_staged_config(
    config: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Metax Modification: prepare staged config for MACA
    stage1_config = config.get("stage1", config).copy()
    stage1_config.setdefault("SPLIT_K", 1)
    stage1_config.pop("ACCF32", None)

    stage2_config = config.get("stage2", config).copy()
    stage2_config.setdefault("SPLIT_K", 1)
    stage2_config.pop("ACCF32", None)
    return stage1_config, stage2_config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int4_w4a16: bool | None = False,
    # ┌------------------------  Metax Modification -------------------------┐
    use_int4_w4a8: bool | None = False,
    use_int8_w8a8: bool | None = False,
    # └------------------------- Metax Modification -------------------------┘
    use_int8_w8a16: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    # ┌------------------------  Metax Modification -------------------------┐
    elif use_int4_w4a8:
        return "int4_w4a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    # └------------------------- Metax Modification -------------------------┘
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        # The output of this function is passed to `try_get_optimal_moe_config`,
        # and as we only simulate OCP MX execution in fused_moe for now,
        # we will NOT look for `*,dtype=w_mxfp4_a_mxfp4.json` for now.
        return None
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def _filter_triton_kernel_config(
    config: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> dict[str, Any]:
    """Drop any stale/unknown keys from tuned configs before Triton launch.

    Some tuned JSON configs may contain historical fields that are not part of
    the Triton kernel signature / launch kwargs. Forwarding them would raise at
    runtime (unexpected keyword argument).
    """

    return {k: v for k, v in config.items() if k in allowed_keys}
