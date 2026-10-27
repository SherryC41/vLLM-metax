#pragma once

// NOTE(metax): Legacy MACA includes share the target CUDA compatibility
// definitions.
#include "core/cuda_compat.h"

// Retain the old shuffle spelling while legacy local kernels still use it.
#ifndef USE_ROCM
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif
