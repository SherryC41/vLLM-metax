# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    MACA_VLLM_USE_TN_2_NN: bool = True
    MACA_DP_OPT: bool = False

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of vLLM, supporting [cuda (by default),
    # rocm, neuron, cpu]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED":
    lambda: bool(os.environ.get("VLLM_USE_PRECOMPILED")) or bool(
        os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, vllm will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),

    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # ================== Runtime Env Vars ==================

    # When installing vllm from source, the version of vllm set by setuptool_scm
    # will be different from the version of vllm installed by pip.
    # (e.g. install vllm from source with tag v0.9.1 will cause the version set
    # as 0.9.2)
    "VLLM_OFFICIAL_VERSION":
    lambda: os.getenv("VLLM_OFFICIAL_VERSION", None),

    # if set, enable loading weight by transpose
    "MACA_VLLM_USE_TN_2_NN":
    lambda: os.environ.get("MACA_VLLM_USE_TN_2_NN", "0") == "1",

    # if set, enable combine allreduce all2all
    "MACA_DP_OPT":
    lambda: bool(os.environ.get("MACA_DP_OPT", 0)),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")