#pragma once
// NOTE(metax): Share stable helpers across legacy and consolidated include
// paths.
#include "libtorch_stable/torch_utils.h"
#ifdef TORCH_TARGET_VERSION
  #define TORCH_UTILS_CHECK STD_TORCH_CHECK
#else
  #include <torch/all.h>
  #define TORCH_UTILS_CHECK TORCH_CHECK
#endif
