# SPDX-License-Identifier: Apache-2.0
import vllm.envs as envs
from vllm_metax import envs as mx_envs

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from pydantic.dataclasses import dataclass

@dataclass
class MacaFusedMoEParallelConfig(FusedMoEParallelConfig):
    
    @property
    def use_combine_allreduce_kernels(self):
        return mx_envs.MACA_DP_OPT and (envs.VLLM_ALL2ALL_BACKEND == "naive" \
                or envs.VLLM_ALL2ALL_BACKEND == "allgather_reducescatter")

import vllm.config.parallel
import vllm.model_executor.layers.fused_moe.config 

vllm.model_executor.layers.fused_moe.config.FusedMoEParallelConfig = MacaFusedMoEParallelConfig
