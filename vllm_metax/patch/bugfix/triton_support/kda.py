# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Limit Triton autotune `num_warps` for KDA kernels on MACA.
#
# Affected versions: vLLM 0.32.0 (5df63a7203cd), reviewed 2026-10-21.
#
# Remove at: Upstream autotuning excludes warp counts unsupported by MACA.
# -----------------------------------------------------------------------------

from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.op import log
from vllm_metax.patch.utils import patch

BT_LIST_AUTOTUNE = [32, 64, 128]

# --------------------------------------
# Note: Maca can't work with nw up to 32
# --------------------------------------
# NOTE(metax): MACA does not support the upstream 32-warp configuration.
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16]


@patch("vllm.third_party.flash_linear_attention.ops.kda", "kda_gate_fwd_kernel")
@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    stride_row = H * D
    stride_col = 1

    o_t = n_t + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    g_ptr = g + i_h * D + o_t[:, None] * stride_row + o_d[None, :] * stride_col
    y_ptr = y + i_h * D + o_t[:, None] * stride_row + o_d[None, :] * stride_col

    b_g = tl.load(g_ptr, mask=m_g, other=0.0).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    if USE_LOWER_BOUND:
        b_y = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
        b_y = -b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), mask=m_g)
