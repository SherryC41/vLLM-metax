# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from typing import Union

import torch

from vllm.distributed import (get_ep_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE


class MacaFusedMoE(FusedMoE):
    
    # ┌------------------------  Metax Modification -------------------------┐
    @property
    def use_combine_allreduce(self):
        return self.moe_parallel_config.dp_size > 1 and \
            self.moe_parallel_config.use_combine_allreduce_kernels
    
    def must_reduce_shared_expert_outputs(self) -> bool:
        return (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels or self.use_combine_allreduce)

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        if (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels or self.use_combine_allreduce):
    # └------------------------- Metax Modification -------------------------┘
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.quant_method is not None

        self.ensure_moe_quant_config()

        # Route to the chunked forward path using the FlashInfer Cutlass kernel
        # only when data parallelism (DP) is enabled.
        _use_flashinfer_cutlass_kernels = (self.dp_size > 1 and
                                           self.use_flashinfer_cutlass_kernels)

        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or _use_flashinfer_cutlass_kernels):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels
            and not self.moe_config.use_flashinfer_cutlass_kernels)

        # If there are shared experts but we are not using a modular kernel, the
        # shared experts must be called here
        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(
            self.sp_size) if ctx.dp_metadata else nullcontext()

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel)

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor,
                              do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    states = get_ep_group().combine(states,
                                                    self.is_sequence_parallel)

                if (not self.is_sequence_parallel and self.reduce_results
                        and (self.tp_size > 1 or self.ep_size > 1)):
                    states = self.maybe_all_reduce_tensor_model_parallel(
                        states)

                return states

            # ┌------------------------  Metax Modification -------------------------┐
            def reduce_shared_output(states: torch.Tensor) -> torch.Tensor:
                if (not self.is_sequence_parallel
                        and (self.tp_size > 1 or self.ep_size > 1)
                        and self.must_reduce_shared_expert_outputs()):
                    states = tensor_model_parallel_all_reduce(states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_shared_output(final_hidden_states[0]),
            # └------------------------- Metax Modification -------------------------┘
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            else:
                return reduce_output(final_hidden_states)

class MacaSharedFusedMoE(SharedFusedMoE):
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            shared_out = self._shared_experts(hidden_states)

            # ┌------------------------  Metax Modification -------------------------┐
            if (self.tp_size > 1
                    and self.must_reduce_shared_expert_outputs()):
            # └------------------------- Metax Modification -------------------------┘
                shared_out = tensor_model_parallel_all_reduce(shared_out)

            fused_out = FusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            shared_out, fused_out = FusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return shared_out, fused_out

import vllm.model_executor.layers.fused_moe.layer
vllm.model_executor.layers.fused_moe.layer.FusedMoE.use_combine_allreduce = MacaFusedMoE.use_combine_allreduce
vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_impl = MacaFusedMoE.forward_impl
vllm.model_executor.layers.fused_moe.layer.FusedMoE.must_reduce_shared_expert_outputs = MacaFusedMoE.must_reduce_shared_expert_outputs
vllm.model_executor.layers.fused_moe.layer.FusedMoE.maybe_all_reduce_tensor_model_parallel = MacaFusedMoE.maybe_all_reduce_tensor_model_parallel

import vllm.model_executor.layers.shared_fused_moe
vllm.model_executor.layers.shared_fused_moe.SharedFusedMoE = MacaSharedFusedMoE