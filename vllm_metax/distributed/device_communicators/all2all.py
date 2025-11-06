# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context

from vllm.distributed.device_communicators.base_device_communicator import All2AllManagerBase

class CoArAll2AllManager(All2AllManagerBase):
    """
    A opt implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = get_forward_context(
        ).dp_metadata.get_chunk_sizes_across_dp_rank()

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
        hidden_states, router_logits = dist_group.all_gatherv(
            [hidden_states, router_logits],
            dim=0,
            sizes=sizes,
        )

        return hidden_states, router_logits

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:

        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = get_ep_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass
