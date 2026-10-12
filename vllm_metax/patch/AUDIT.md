# Runtime patch audit — 2026-10-12

Reviewed target: vLLM **0.30.0** (`ced6857afa0e`); tokenizer patches also
reviewed against Transformers **5.19.0**. Patch paths below are relative to
`vllm_metax/patch/`; upstream paths are relative to `vllm/`.

## Activation

`vllm_metax:register_out_of_tree` calls `_patch()` before `_out_of_tree()`.
`_patch()` imports `bugfix`, `enhancement` and `performance`, activating their
child modules; `_out_of_tree()` loads the ordinary registries used by migrated
implementations. Entries below are active unless marked removed or migrated.

## Bugfix patches

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `bugfix/bytes_to_unicode.py`: `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode` | **Retain.** Transformers 5.19 still lacks the GPT-2 export imported by affected remote tokenizers. Adds it only when absent, with caching. Remove when affected tokenizers stop importing the removed export. |
| `bugfix/tokenizer.py`: `LlamaTokenizerFast.convert_to_native_format` | **Retain.** Definition is inherited from `tokenization_utils_tokenizers.TokenizersBackend`, not removed. The 5.19 reconstruction branch says it preserves the decoder/pre-tokenizer, but does not actually return either in its kwargs. Select the full `tokenizer_object` branch only for an existing tokenizer.json; preserve native conversion otherwise. The branch-selection flag executes no remote code. Remove when the reconstruction preserves complete backend semantics. |
| `bugfix/draft_config_overrides.py`: `SpeculativeConfig.compose_draft_hf_overrides` | **Retain.** `config/speculative.py:1064` still deliberately leaves dictionary overrides target-only; callable overrides are composed upstream. MetaX adds picklable dictionary inheritance and nested `PretrainedConfig` updates; other cases delegate. Remove when upstream provides dictionary inheritance for these checkpoints. |
| `bugfix/deepseek_v4/parallel_state.py`: `GroupCoordinator._should_use_all_gather` | **Retain.** `distributed/parallel_state.py:1053` still chooses the PP optimization by divisibility and optional overrides. MetaX disables it to avoid the recorded MACA trap. Remove only after suitable PP/TP hardware validation proves that path safe. |
| `bugfix/int8_w8a8/int8_moe_config.py`: `int8_w8a8_moe_quant_config` | **Retain.** `model_executor/layers/fused_moe/config.py:634` still lacks SwiGLU alpha/beta/clamp forwarding in W8A8; W8A16 already forwards it. Only the W8A8 extension remains. Remove when upstream forwards these three fields. |
| `bugfix/int8_w8a8/mla_chunked_prefill.py`: `_get_kv_b_proj_input_dtype` | **Retain.** `model_executor/layers/attention/mla_attention.py:379` returns INT8 for INT8 weights; MetaX's linear method dynamically quantizes model-dtype input. Intercept only INT8 and delegate INT32, packed UINT8, ModelOpt, FP8 and ordinary types. Remove when upstream supports this input contract. |
| `bugfix/qwen3_5_moe_loading.py`: `AutoWeightsLoader.load_weights` | **Retain.** `model_executor/models/utils.py` preserves quantization mappers and reload support but does not dequantize the affected ignored MTP gates. The wrapper delegates to that decorated original and only transforms the exact Qwen3.5 MTP class's weights. Remove when checkpoints or upstream loading resolve this mismatch. |
| `bugfix/telechat3/telechat3_yarn.py`: `_get_and_verify_max_len` | **Update.** `config/model.py:2393` treats YaRN lengths as already scaled and filters None entries for layers without RoPE. Keep temporary telechat3-yarn normalization, preserve nested None entries, and retain explicit length validation/encoder overrides. Remove when upstream directly recognizes telechat3-yarn. |
| `bugfix/triton_support/kda.py`: `kda_gate_fwd_kernel` | **Retain.** `third_party/flash_linear_attention/ops/kda.py` still includes 32 warps for non-AMD platforms. Body is unchanged; MACA autotune omits that unsupported count. Remove when upstream/MACA supports a safe native configuration. |
| `bugfix/triton_support/chunk_delta_h.py`: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | **Retain.** Body/heuristics match `third_party/flash_linear_attention/ops/chunk_delta_h.py`; upstream stages differ. Keep single-stage tuning for the recorded MACA shared-memory restriction. Remove when native launches fit this hardware limit. |
| `bugfix/minimax_m3/index_topk.py`: `minimax_m3_index_score`, `minimax_m3_index_decode_score` | **Update activation; retain launch adjustments.** `models/minimax_m3/common/ops/index_topk.py` still needs single-stage launches and the minimum 16-row MMA query tile on MetaX. Patch the defining exports once; the default installer synchronizes loaded `common.ops` and `common.indexer` aliases, while later imports bind the replacement normally. Remove when native launches satisfy MACA resource/tile constraints. |
| `bugfix/minimax_m3/sparse_attn.py`: `_gqa_sparse_fwd_kernel`, `_gqa_sparse_decode_kernel` | **Retain.** `models/minimax_m3/common/ops/sparse_attn.py` still loads 128-token K/V tiles. Keep SUB_K=16 and prefill head-tile floor; scale strides, causal masks, padded-row clamping, split-K empty-chunk handling and PDL behavior remain inherited. Remove when native tiling fits MACA or its limit is resolved. |
| `bugfix/minimax_m3/load_weights.py`: `MiniMaxM3Model.load_weights` | **Update.** `models/minimax_m3/nvidia/model.py` unconditionally renames inverse scales for MXFP8. Preserve inverse scales only if direct/fused/expert destination parameters expose that name; otherwise inherit the upstream rename. All other loading logic remains current. Remove when upstream chooses scale naming from actual destination parameters. |

## Enhancement patches

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `enhancement/chores/bench_serve_args.py`: both `add_cli_args` exports | **Retain.** `benchmarks/serve.py` still defaults temperature to None; `entrypoints/cli/benchmark/serve.py` binds its alias. MetaX wrapper defaults to 0.0 while preserving explicit settings and the full argument set. Remove when benchmark policy adopts the upstream default. |
| `enhancement/chores/maca_visible_device.py`: `WorkerWrapperBase.update_environment_variables` | **Retain.** `v1/worker/worker_base.py` only installs supplied variables. Copy the selected RPC dictionary and mirror CUDA visibility only when supplied; explicit MACA values win. Remove when upstream propagates MACA visibility with these semantics. |
| `enhancement/chores/remove_error_log.py`: `get_flash_attn_version` and loaded aliases | **Update activation; retain FA selection.** `v1/attention/backends/fa_utils.py:74` probes NVIDIA versions and still has the same seven-argument interface. Rebind preloaded callers, including `model_executor/layers/attention/mm_encoder_attention.py`. Remove when the platform hook handles MetaX selection natively. |
| `enhancement/distributed/cuda_wrapper.py`: `cudaIpcMemHandle_t`, `CudaRTLibrary`, loaded class aliases | **Update activation; retain MACA bindings.** `distributed/device_communicators/cuda_wrapper.py` lacks MACA symbols and uses the NVIDIA IPC layout. `all_reduce_utils.py` bound the old class before patch activation; rebind it. Remove when native bindings support MACA and its 128-byte handle. |
| `enhancement/distributed/pynccl_wrapper.py`: both `NCCLLibrary` exports | **Retain.** Upstream `pynccl.py` and `pynccl_wrapper.py` still require replacements for MCCL symbols. Patch the defining `pynccl_wrapper` export once; default alias synchronization also updates loaded `pynccl` callers. Remove when upstream dispatches natively to MCCL. |
| `enhancement/distributed/utils_patch.py`: `find_nccl_library`, `import_pynvml`, loaded aliases | **Update activation; retain MCCL/MX-SMI discovery.** Upstream `utils/nccl.py` and `utils/import_utils.py` do not dispatch to these MetaX libraries; the loaded `pynccl_wrapper.find_nccl_library` previously remained stale. Remove when native discovery supports MetaX. |
| `enhancement/dbo.py`: `SMControlContextManager.__init__` | **Update.** `v1/worker/ubatch_utils.py` still only asserts CUDA/ROCm; keep the CUDA-like check, fix the erroneous call of its boolean return, and preserve resource checks/callback fields. Remove when upstream accepts CUDA-like OOT platforms. |
| `enhancement/joyai_support/joyai_transformer_config.py`: converter registry entry | **Retain.** `transformers_utils/model_arch_config_convertor.py` still omits `joyai_llm_flash` from MLA detection. Adds only that converter; inherited conversion behavior remains upstream. Remove when upstream recognizes the architecture. |
| `enhancement/quant_kernels/fp8_block_kernel.py`: OOT FP8-block candidate table | **Retain.** `model_executor/kernels/linear/__init__.py:1189` still has no public `fp8_block` registration type. Append only the MetaX candidate without replacing other platforms. Remove/migrate when the public API supports this table. |
| `enhancement/quant_kernels/mxfp8_kernel.py`: OOT MXFP8 candidate table | **Migrate.** The public `register_linear_kernel(..., "mxfp8")` branch exists in the target. Delete this private-table patch/import and register the same candidate in `registry/linear_kernels/mxfp8.py`, loaded by that registry's initializer. No monkey patch remains; registration is still necessary for OOT. |
| `enhancement/deepseek_v41_support.py`: package export `DeepseekV4FP8Config` | **Remove.** `model_executor/layers/quantization/__init__.py:124` reads the V4.1 package export only for CUDA/ROCm; MetaX is OOT and takes the V4 branch. Existing `platform.py` instead registers the MetaX V4.1 config via the supported quantization registry, whose overrides win at line 194. The package-only assignment did not update the defining module or already-imported classes. Remove the redundant activation import too. |
| `enhancement/utils.py`: `_int8_quantize` | **Retain.** `model_executor/layers/fused_moe/utils.py:249` still uses the Triton per-token quantizer. Substitute MetaX scaled INT8 quant only in that branch; intentionally ignore supplied static scale for dynamic per-token mode. Remove when native INT8 dispatch supports MetaX equivalently. |
| `enhancement/utils.py`: `FusedMoEQuantConfig.use_int4_w4a8` | **Retain.** The property is absent upstream and used by MetaX W4A8 experts. This intentionally added attribute is the only class target using `allow_missing=True`. Remove when the descriptor is supported upstream or no longer used. |
| `enhancement/utils.py`: `moe_kernel_quantize_input` / aliases | **Update shared alias installation; retain dtype extension.** Upstream handles `torch.int8` but lets string `"int8"` pass through unquantized. Keep the narrow string branch and delegate every other dtype/argument; rely on default identity-based alias synchronization without per-file helper calls, `allow_missing`, or swallowed installation failures. Remove when upstream supports the string descriptor and MetaX quantizer. |

## Performance patches

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `performance/grouped_topk_router.py`: `GroupedTopKRouter._compute_routing` / MetaX helpers | **Retain.** `model_executor/layers/fused_moe/router/grouped_topk_router.py` routing method matches outside MetaX dispatch; its fallback grouped algorithm matches scoring/bias/scaling/batch-invariance behavior. Keep native MetaX fused routing and SGL shape policy. Remove when native router dispatch supports these kernels. |
| `performance/speculative_decode_perf.py`: `GPUModelRunner._may_reorder_batch` / reorder helper | **Retain.** `v1/worker/gpu_model_runner.py:1138` delegates to the same four-region algorithm in `v1/attention/backends/utils.py:882`. MetaX additionally stably sorts completed-prefill decodes by query length when speculative decoding and backend grouping are enabled; ordinary paths delegate upstream. Remove when native reordering supports required decode buckets. |

## Shared infrastructure

| Module | Purpose and repair reason | Removal condition |
| --- | --- | --- |
| `utils.py` | Installs module/class replacements, preserves descriptors and records originals. Updating a defining module alone leaves previously imported aliases stale, so module patches also synchronize same-name exports in loaded `vllm.*` modules by object identity. Automatic aliases may become explicit stacked targets once; repeated explicit or conflicting replacements are rejected. The alias-target lookup uses `set[str]` with a set fallback so `remove()` has a consistent type. Class patches update their shared owner without scanning bound-method copies. | All runtime patches migrate to native extension points or another installer. |
| `__init__.py` | Exposes the shared `patch` decorator; does not install patches itself. | No module imports this package-level export. |
| Child package `__init__.py` files | Activate their child patches by import. Removed and migrated implementations no longer appear in these imports. | Their child patches have all been removed or migrated. |
| `template/class_method.py`, `template/module_attr.py` | Inactive examples of descriptor-preserving and module-attribute patches, including default alias synchronization and Triton decorator order. | The runtime patch mechanism is no longer needed. |

Alias synchronization preserves MetaX's captured originals, unrelated objects
with the same name, renamed exports and references in local variables or
containers. Original metadata is recorded without recursively invoking
`patch()`. The installation flow and alias boundaries are documented in
[README.md](README.md).
