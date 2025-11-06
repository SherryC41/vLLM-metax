# SPDX-License-Identifier: Apache-2.0
import torch

import vllm.envs as envs
from vllm_metax import envs as mx_envs

from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

class MacaFusedMoE(FusedMoE):
    
    @property
    def use_combine_allreduce(self):
        return self.moe_parallel_config.dp_size > 1 and mx_envs.MACA_DP_OPT \
            and (envs.VLLM_ALL2ALL_BACKEND == "naive" \
                or envs.VLLM_ALL2ALL_BACKEND == "allgather_reducescatter")
    
    def must_reduce_shared_expert_outputs(self) -> bool:
        return (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels or self.use_combine_allreduce)

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        
        if (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels or self.use_combine_allreduce):
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)
        
import vllm.model_executor.layers.fused_moe.layer
vllm.model_executor.layers.fused_moe.layer.FusedMoE = MacaFusedMoE