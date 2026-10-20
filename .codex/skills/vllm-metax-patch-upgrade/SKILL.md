---
name: vllm-metax-patch-upgrade
description: Audit and adapt monkey patches in vllm_metax/patch/ against a target upstream revision. Decide whether to retain, update, migrate, or remove each patch; maintain patch headers and explanations; validate changes and record evidence. Use only for compatibility work on this directory, not standalone attention backend, model, kernel, or other adaptations elsewhere in vllm_metax.
---

# vLLM-MetaX Patch Upgrade

Read [vllm-metax-upgrade-common](../vllm-metax-upgrade-common/SKILL.md) in full and
apply it before this specialist workflow.

## Scope

Apply this skill only when the requested adaptation concerns `vllm_metax/patch/`.
An implementation adapted from upstream is not automatically a monkey patch in
this scope. Standalone work on attention backends, models, kernels, or other
directories must use the normal repository workflow, without inheriting this
skill's patch headers or patch audit requirements.

For an in-scope patch, inspect upstream code, callers, registrations, and tests
outside the directory as needed. Changes there must directly support that patch's
adaptation, removal, migration, or validation. If a request spans both patch and
non-patch work, apply this skill only to the patch portion; keep unrelated backend
or model findings out of `vllm_metax/patch/AUDIT.md`.

Read `vllm_metax/patch/README.md`, patch templates and existing audit records before
working on the patch inventory.

## Inventory patches and activation paths

Start from the `vllm_metax/patch/` file list and trace plugin entry points and package
initializers. Search beyond `@patch`: include direct attribute assignments, registry
mutations, `sys.modules` redirects, imported aliases, and disabled or unimported patches.

For each patch, record its file, target symbol, purpose, activation status, upstream
location, MetaX differences, decision, evidence, and validation approach. A single
file may contain patches with different decisions. Classify templates, shared helpers,
and aggregate import modules separately as infrastructure.

## Decide per patch

Read the original patch explanation and resolve its actual upstream target, including
dynamic exports, inherited methods and imported aliases. A failed static lookup does
not establish that the target was removed.

For each disposition, account for the replacement, activation entries and bound callers:

- Removal must retire activation entries and prove callers reach the intended original
  or existing redirect instead of a stale alias.
- Migration must activate the supported hook/registry/CustomOp and remove overlapping
  old replacements after checking caller bindings.
- Retained or updated patches must still bind the exact intended target under the
  plugin's current import/initialization order.

- Age, code similarity, disabled status, or an existing PR link alone is not evidence
  for removal. Inspect history when needed and confirm the fix exists in the target revision.
- Investigate why upstream deliberately removed old behavior before restoring it
  from a patch's copied implementation.
- For kernel/communication patches, check the specific shared-memory, warp, IPC ABI
  or communication restriction behind retention; an unchanged kernel body is insufficient.
- For ordering patches, verify stability over the whole claimed region; sorting only
  misplaced requests does not establish regional stability.

## Apply the patch decision

- Prefer existing upstream registration or extension points. A small wrapper should
  delegate unaffected behavior to the original implementation and preserve explicit inputs.
- Check full replacements for target validation, model branches and cleanup that an
  old patch copy could suppress.
- Preserve descriptor semantics, Triton decorator order, lazy imports, and initialization
  timing. Check whether `from ... import ...` already bound the old object; successful
  patch import does not prove every caller uses the replacement.
- Follow repository templates for `@patch`. Use `allow_missing=True` only when adding
  an intentional compatibility attribute, never to hide renamed targets or typos.
- Inspect argument semantics rather than forwarding by name alone. For example,
  `scale=None` may select dynamic quantization while a supplied scale selects static quantization.
- Update only the platform's registry entries rather than replacing the entire table.
  Check registration priority and repeated activation where relevant.
- On removal, check references, import order and overlapping replacements.

For quantization, tokenizer, batch ordering, registration, allocator, or kernel patches,
consult [review-patterns.md](references/review-patterns.md) as needed. Its examples guide
inspection; they are not fixed decisions for future upstream versions.

## Preserve patch headers and explanations

Follow the current patch README. Every Python file in the patch directory, including
initializers and shared utilities, should have the dedicated header. Preserve license
and copyright notices. Templates may retain fill-in placeholders; active files must
contain meaningful descriptions.

```python
# -----------------------------------------------------------------------------
# Note: Describe the concrete issue, trigger, and reason for the MetaX difference.
#
# Affected versions: State evidence-backed affected versions and the reviewed revision.
#
# Remove at: Give a verifiable condition, such as upstream support or a resolved limit.
# -----------------------------------------------------------------------------
```

- Do not substitute scattered `Verified against`, `Remove when`, or docstrings for
  these fields. Describe infrastructure honestly without inventing an upstream defect.
- For complex sorting or kernel patches, explain input/output contracts, invariants,
  key indices and mappings, the rationale, and the actual differences from upstream.
- Distinguish correctness requirements, platform restrictions, performance policy, and
  implementation choices. Preserving non-decode request order does not mean one specific
  request must precede another for numerical correctness.
- Include a discriminating input/output example for the patch's boundary behavior.

## Verify patch application and callers

- Run patch application in fresh processes to avoid duplicate-patch failures. Verify
  both the target import and that actual callers reach the replacement, including
  callers which previously bound the original with `from ... import ...`.
- For configuration patches, exercise explicit and nested overrides, repeated calls
  and serialization. For batch-order patches, check request/state alignment through
  reordering. For quantization/kernel patches, cover affected dtypes, masks and layouts.

## Update the patch audit

Update the existing `AUDIT.md` or repository-designated patch review record. Account
for every inventory entry and ensure decisions match the final code. Include each
retain/update/migrate/remove decision, concrete target evidence, removal condition,
activation/caller verification and the important behavioral differences. Link the
shared environment and validation evidence rather than duplicating their full records.
Keep unrelated domain findings outside this patch audit.
