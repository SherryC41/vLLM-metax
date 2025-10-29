# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm_metax import envs as mx_envs
from vllm.logger import init_logger
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

logger = init_logger(__name__)


class MacaCommunicator(CudaCommunicator):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        
        if self.use_all2all:
            all2all_backend = envs.VLLM_ALL2ALL_BACKEND
            if mx_envs.MACA_DP_OPT and (all2all_backend == "naive" \
                or all2all_backend == "allgather_reducescatter") :
                from .all2all import CoArAll2AllManager
                self.all2all_manager = CoArAll2AllManager(
                    self.cpu_group)
                logger.info("Using combine_allreduce manager for opt.")

import vllm.distributed.device_communicators.cuda_communicator

vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = MacaCommunicator