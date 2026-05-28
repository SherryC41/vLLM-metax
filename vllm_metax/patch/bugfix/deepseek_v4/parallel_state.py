# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------
# Note: This patch is fix DeepSeek-V4 trap when use pipeline parallel,
#       temporarily disable the optimization of use all_gather in send/recv
# ------------------------------------------------------------

from vllm.distributed.parallel_state import GroupCoordinator


def _should_use_all_gather(
    self,
    key: str,
    numel: int,
    all_gather_group: "GroupCoordinator | None",
    all_gather_tensors: dict[str, bool] | None,
) -> bool:
    return False


GroupCoordinator._should_use_all_gather = _should_use_all_gather
