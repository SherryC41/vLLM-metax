---
name: vllm-metax-upgrade-common
description: Shared workflow for vLLM-MetaX compatibility adaptations and reviews, including confirmed environments, upstream-first code, dependency contracts, validation and durable progress. Required by standalone domain skills and the full-upgrade coordinator; collecting environment facts may precede confirmation.
---

# vLLM-MetaX Upgrade Common

Apply this complete workflow before the calling skill's domain-specific instructions.
It owns the shared rules; specialists define only their scope, contracts and cases.
Do not activate unrelated specialists for standalone work. Maintaining these skills
or their tooling does not itself start a compatibility upgrade.
Patch headers and `vllm_metax/patch/AUDIT.md` apply only to actual patch-directory
work; copying an upstream implementation elsewhere does not make it a monkey patch.

## Scope, review subject and ownership

Identify the requested scope and whether the subject is the index, working tree,
installed package or a specified revision. Read applicable repository instructions.
Preserve existing user changes. Keep reviewed source read-only; durable environment,
progress and review/evidence artifacts are allowed. Do not stage, commit, reinstall,
change global import paths or modify upstream/site-packages without authorization.
For staged validation, export the index
and prove the actual test imports that snapshot; unstaged fixes cannot supply its
missing dependencies. Justified import overrides must remain process-local.

Locate the existing durable run before creating one. Read its state, TODO, baseline,
last handoff and source changes. Standalone work uses component mode; coordinated
work uses the coordinator's run, assigned unit and contracts. Do not create a second
baseline or edit another owner's shared unit. Return discovered dependencies and
evidence to the coordinator. Specialist selection never expands the user's scope.

## Mandatory environment gate

Read [environment-preflight.md](references/environment-preflight.md) for collection,
source/runtime correspondence and confirmation. Use the selected validation Python
and the independent stdlib inventory in `tools/metax_upgrade/environment.py` to dump
SDK/driver/GPU, Python/environment, installed packages, `maca_private.txt` comparisons
and released-target/source/runtime mappings. Each skill request collects fresh facts;
historical reports or another skill's output cannot replace current collection.
The selected MetaX `collect_env.py` is an optional plain-text supplement, without CLI
arguments or a structured API dependency; its failure must not block independent inventory.

**Present the collected snapshot and fingerprint and obtain the user's explicit
confirmation before compatibility decisions, adaptation edits or runtime validation.**

Before confirmation, only environment probes and independent read-only inventory may
run. Never fabricate confirmation or infer it from silence or successful collection.
If pausing, link this SKILL.md, quote the gate and present a self-contained environment
table, issues and snapshot/TODO paths. Do not install or change the environment as
part of collection. Recheck current identity/mapping on resume; unchanged facts may
retain the actual prior approval, while changed selections require a new confirmation.
Expected adaptation edits invalidate source evidence without forcing environment reapproval.

## Inventory and dependency contracts

- Fix the confirmed released vLLM revision as the upstream contract. All MetaX modules,
  unchanged code, old wheels, startup helpers and tests begin unverified. Comments,
  names, signatures and upstream support declarations are claims to check, not proof
  of MACA capability. Installed components need their own API/ABI and behavior evidence.
- Inventory every requested file/symbol, including retained, removed, conditional and
  inactive adaptations. Record the exact upstream counterpart, prior MACA semantics,
  callers, differences, disposition, reason/removal condition and required validation.
  Choose retain, update, migrate, remove or explicitly unsupported; a passing subset
  does not cover the requested inventory. Persist file-level target/prior-source mappings
  and MACA reasons as checked mapping artifacts, not only prose notes. Platform-only
  and removed units need explicit dispositions rather than invented upstream symbols.
- Establish the previous upstream base and confirmed target, mapping moved, local-only
  and target-only files explicitly. For a known upgrade range, inspect every upstream
  commit touching the mapped units, other production files changed in those commits
  and direct callers/dependencies. Record inherited/no-edit decisions as well as merges,
  coordinated changes and unsupported additions. Commit titles or the final tree diff
  alone cannot account for the upgrade's semantic changes.
- Trace reachable imports, startup side effects, registrations, calls and data/state
  contracts, including conditional paths and transitive consumers of changed interfaces.
  Record argument/default meanings, returns, lifecycle, dtype/layout/index/scale units,
  ownership and side effects as applicable. Recurse to a verified stable interface or
  installed component boundary and state why that boundary is sufficient.
- Adapt dependent interfaces together when necessary. Unfinished dependencies can
  block validation while independent source work continues; record the precise blocked
  obligation and resume condition. Do not semantically import unfinished MetaX code
  into the coordinator or use its results as an independent reference. Candidate
  execution after the gate belongs in fresh subprocesses with declared boundaries.

## Upstream-first implementation

- Inspect and preserve the old adaptation before replacing it. Start from the complete
  exact target module/function text, then reapply only verified, still-required
  MACA/MetaX semantics. An old workaround is evidence to investigate, not automatic truth.
- Preserve upstream organization, names, function/class boundaries, method/parameter
  order, decorators, control flow, comments and writing style. Map directory moves and
  parameter additions/removals explicitly; the module/function retains the upstream
  semantic role and required platform behavior. Keep explicit user settings and
  caller-owned state intact unless the intended upstream contract requires a change.
- Do not optimize, refactor, rename, extract helpers or broadly reformat during the
  upgrade. Review both the local change diff and complete target-vs-MetaX diff; justify
  every remaining difference in the touched unit, including retained differences.
- Mark required differences with concise nearby `NOTE(metax):` reasons, for example
  `# NOTE(metax): MCOPLIB expects BF16 scales.` One note may cover a coherent adjacent
  change. Use native comment syntax where appropriate; for formats without comments,
  put the note in the mapped task/audit. Preserve useful algorithm/layout explanations
  and any additional headers required by the specialist.

## Validation, progress and reporting

Read [validation.md](references/validation.md) before planning checks or attributing
defects. It defines isolated checks, evidence boundaries, actual runtime provenance,
shared-upstream attribution and re-review. Specialists supply the discriminating
domain cases instead of redefining this policy.

Read [run-state.md](references/run-state.md) before work begins. Use
`tools/metax_upgrade/upgrade.py` for the authoritative state and generated TODO in a
durable run directory. Persist task scope, upstream mapping, contracts, dependencies
and acceptance obligations before edits; checkpoint meaningful discoveries/results
and the next executable step before yielding or an expected quota stop. On interruption,
reconcile actual hashes and partial diffs rather than trusting existing files or prose.

A source change is `implemented` while required evidence remains missing. Requeue
blocked checks when dependencies are ready; changes to source, contracts, environment,
component identity, validation inputs or tool/rule versions invalidate affected evidence
and consumers. Freeze per-task numerical references, seeds, tolerances, special-value
rules and required cases before running numerical checks; retain measured case results. `done` requires
current evidence for the declared configuration, not merely a successful command.
Persist support-matrix rows and their configuration/evidence obligations for full runs;
every supported row requires its own normal-entry e2e evidence. Numerical and production
checks also need a configuration-matched row in component runs. Resume old run schemas
through migration and explicitly reconcile tool/rule changes before new checks; preserve
prior state and stale evidence. These workflow acknowledgments do not replace the
user's environment confirmation.

Report concrete changes/findings, triggers and consequences, dispositions, verified
scope, limitations and reproducible results. Include the target/approved environment,
run/TODO paths, completed/remaining/blocked work and next action when incomplete. Use
the user's audit location; no separate audit file is required unless the specialist
or repository requires one. Full-repository completion requirements belong to the
coordinator; component completion makes only its declared domain/configuration claim.
