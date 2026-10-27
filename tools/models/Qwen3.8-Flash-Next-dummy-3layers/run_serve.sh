#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Choose idle GPUs explicitly. TP4/TP8 are dimensionally compatible candidates,
# subject to installed kernels and actual free memory; neither is GPU-validated.
: "${GPU_IDS:?Set GPU_IDS to idle GPUs, for example GPU_IDS=0,1,2,3}"
IFS=',' read -r -a selected_gpus <<< "$GPU_IDS"
TP="${TP:-${#selected_gpus[@]}}"
if [[ "$TP" != "${#selected_gpus[@]}" ]]; then
    echo 'TP must equal the number of selected GPUs for this single-node script.' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS" MACA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
extra_args=()
if [[ "${MTP:-0}" == 1 ]]; then
    extra_args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi
exec vllm serve "$MODEL_DIR" \
    --served-model-name qwen-flash-dummy \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --load-format dummy --dtype bfloat16 \
    --tensor-parallel-size "$TP" --pipeline-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-seqs 1 --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    --no-enable-prefix-caching --enforce-eager \
    "${extra_args[@]}" "$@"
