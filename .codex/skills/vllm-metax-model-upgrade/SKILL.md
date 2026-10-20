---
name: vllm-metax-model-upgrade
description: Review and upgrade MetaX model support against target vLLM and installed MACA components, including model-dependent attention, weight loading and kernels. Use for model adaptations and shared-upstream-bug attribution, not standalone patch or registry audits.
---

# vLLM-MetaX Model Upgrade

Fully read and apply [vllm-metax-upgrade-common](../vllm-metax-upgrade-common/SKILL.md) before this workflow.

Cover the requested models under `vllm_metax/models/`, their registration/configuration,
and required attention, cache, compressor, indexer, projection, MTP/draft, MoE and
kernel wrappers. Model-private attention remains in scope wherever its files live.

## Establish the MACA/upstream difference matrix

Build the matrix for each requested model family using the effective configuration
after platform and speculative-decoding rewrites, not only the raw HF config.

- Inspect the exact checkpoint variant/revision's official `config.json` for layer
  layout. Record backbone and MTP/draft counts separately, count layer-indexed arrays
  such as `layer_types`, `rope_theta` and `partial_rotary_factors`, and map zero-based
  checkpoint IDs to each part. Config defaults and the highest layer ID do not supply
  the layer count; identify missing official configuration as an evidence gap.
- Verify the config class selected by target vLLM. Compare raw checkpoint values with
  the constructed configuration after overrides, truncation and other rewrites before
  model/MTP code indexes layer arrays.

| Contract | Record for both upstream and MACA |
| --- | --- |
| Weights | Checkpoint/runtime dtype, per-channel/block/MX grouping, scale encoding, packing, post-load conversion and excluded modules. |
| Activations and projections | Compute/output/accumulation dtype, fused/separate projections, grouped GEMM, padding, quantization and inverse RoPE order. |
| KV and indexer caches | Logical/physical dtype, quantization scales, compressed/SWA separation, page layout/stride and index units. |
| Attention | Actual backend/model-private path, prefill/decode/MTP, head dimensions, sinks, causal behavior, compression and capability restrictions. |
| MoE and parallelism | Actual experts implementation, routing/scaling, shared experts, TP/PP/SP/EP/DCP collective ownership and supported combinations. |
| Optional features | LoRA, MTP/DSpark, graph capture and each feature's actual dispatch and restrictions. |
| Components | Selected DeepGEMM, FlashAttention/FlashMLA, MCOPLIB and other callable contracts and verified semantics. |

A model's FP8/FP4 checkpoint label does not replace required MACA BF16/INT8 compute,
weight or cache paths. Inspect the model's executable selection and conversion rather
than deriving these paths from its checkpoint label.

## Follow model-dependent interfaces

Trace model registration, config rewriting, layer construction, weight loading,
execution selection and kernels. Include model-private attention/compressor/FlashMLA
implementations such as DeepSeek V4 even outside a generic attention directory.

Check config transforms and model/draft class selection alongside parameter mappings;
identically named imported classes can bind to different local/upstream implementations.
Verify binary API contracts at the installed boundary. Read relevant
[review-cases.md](references/review-cases.md) for quantization/loading, private attention,
parallelism and model-identity cases; its historical examples are investigation prompts.

## Bound repairs to shared upstream model bugs

When a defect also affects or may affect target upstream:

- Prefer the smallest model-local fix/workaround that preserves the upstream model role
  and required MACA behavior. Its inline difference note must identify the shared
  upstream scope and why the local workaround is needed; retain the evidence and
  removal condition.
- If repair requires changes to other modules/components, keep that shared-bug repair
  outside this model adaptation unless separately authorized. Add a targeted
  `logger.warning` or `logger.warning_once` with trigger, consequence and known workaround,
  and report the unresolved repair plan. Prefer construction/configuration time to
  repeated token-time logging.
- Keep existing rejection/capability checks: the warning cannot enable an unsupported
  path or replace an existing error with silent success.
- This boundary applies to shared-bug repair, while dependency updates required by the
  requested model's changed interface/semantics remain part of the upgrade.

## Model acceptance cases

Verify registration and construction, real checkpoint loading, transformed parameter
state, selected model/kernel paths and the affected numerical behavior. Derive boundary
and rejection cases from the difference matrix, including loader ordering and cache
scale/packing cases.

For parallel changes, verify collective ownership and residual semantics through the
actual distributed path. LoRA/MTP support requires their model execution paths, and
wrapper-only or kernel-only results cannot establish that support. State unresolved
shared-bug warnings alongside the model-specific difference and dependency coverage.
