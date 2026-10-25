# MetaX upgrade run tools

Use the repository skill `$vllm-metax-upgrade` for a full upgrade, or a specialist
skill for a limited component. These standard-library tools persist the environment
gate, worklist and evidence; the agent still reviews contracts and implements code.
They do not run an autonomous model service or install dependencies.

The project custom agent is `.codex/agents/vllm_metax_upgrade.toml`, named
`vllm_metax_upgrade`. Ask Codex to use that agent, or invoke the top-level skill
directly. It inherits the session's model and permissions. Its project file follows
the [official custom-agent format](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents).
Another AI can follow the same skill and resume the durable run without this role.

Invoke every command with the **selected validation interpreter** from the repository
root. For example, replace the interpreter and target paths below as needed:

```bash
/opt/venv/bin/python tools/metax_upgrade/upgrade.py init \
  --run-dir .metax-upgrade/release-upgrade \
  --vllm-source /root/workspace/vllm \
  --metax-source /root/workspace/vLLM-metax \
  --runtime-cwd /root/workspace/vLLM-metax \
  --scope 'Requested models, quantization and parallel configurations' --mode full
```

The command only prepares a run. Review `ENVIRONMENT.json`, the immutable JSON
under `environment/`, the original text in `ENVIRONMENT.txt`, and `TODO.md` with the
user. `tools/metax_upgrade/environment.py` independently collects SDK/GPU identities,
the selected interpreter's package metadata, `requirements/maca_private.txt`
comparisons, source/runtime correspondence and native-file identities. It does not
import or require a structured API from `vllm_metax/collect_env.py`.

The normal collector keeps its original `main -> get_pretty_env_info -> pretty_str`
flow and section style, takes no command-line options, and prints all installed
packages by default. Use shell redirection to save it:

```bash
vllm_collect_env > environment.txt
# Or use an explicitly selected interpreter and checkout:
/opt/venv/bin/python vllm_metax/collect_env.py > environment.txt
```

The toolkit optionally runs that plain script in a child with automatic plugin
activation disabled and records its sanitized text, command, timestamp and status.
Each full collection runs the script afresh; specialists collect their own current
facts rather than reusing another skill's output. Saved reports remain historical
evidence, not a source of live environment data. `refresh --run-dir ...` collects a
new full snapshot. Collector failures retain the independently collected environment
facts and their limitations; they do not establish compatibility or silently approve
anything. Native stat/RECORD identities likewise do not prove ABI or numerical support.

Only after an explicit user reply confirming the displayed snapshot:

```bash
/opt/venv/bin/python tools/metax_upgrade/upgrade.py confirm \
  --run-dir .metax-upgrade/release-upgrade \
  --fingerprint HASH_FROM_ENVIRONMENT_REPORT \
  --user-response-file .metax-upgrade/release-upgrade/user-reply.txt
/opt/venv/bin/python tools/metax_upgrade/upgrade.py assert-ready \
  --run-dir .metax-upgrade/release-upgrade
```

Save the actual user message to `user-reply.txt`. Add `--acknowledge-issues` only if
the user explicitly accepts the disclosed missing facts or mismatches. The tool records
the reply; it cannot authenticate the speaker or determine that the wording grants
approval. The skill requires the agent to check this. No confirmation is fabricated.

Before modifying a unit, register its files, configuration and required evidence:

```bash
/opt/venv/bin/python tools/metax_upgrade/upgrade.py task-add \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout \
  --title 'Adapt layout contract' --files vllm_metax/actual_module.py \
  --requires static --requires review --requires component_numeric --requires closure --requires e2e \
  --configuration bf16-single-gpu --next-action 'Compare the exact upstream function'
```

Repeat `--files`, `--requires` and `--depends-on` as needed. Dependencies name existing
tasks; combine cycles into a coordinated task. Use one task per exact configuration.
Every task needs a checked mapping before it can become `done`. Register complete
file coverage with `mapping-put --id TASK --file mapping.json`; task notes alone do
not satisfy this requirement. A mapping document has this shape (replace all example
paths, revisions and hashes with actual source identities):

```json
{
  "mappings": [{
    "local": "vllm_metax/actual_module.py",
    "upstream": {
      "revision": "TARGET_REVISION",
      "path": "vllm/path/to/module.py",
      "symbol": "actual_function",
      "sha256": "ACTUAL_TARGET_CONTENT_SHA256"
    },
    "previous": {
      "revision": "PREVIOUS_UPSTREAM_REVISION",
      "path": "vllm/path/to/module.py",
      "symbol": "actual_function"
    },
    "disposition": "update",
    "metax_semantics": "State the verified previous MACA behavior to preserve",
    "reason": "Explain why this target still needs that difference",
    "removal_condition": "State the concrete condition for removing the difference"
  }]
}
```

Dispositions are `retain`, `update`, `migrate`, `remove`, `unsupported` and
`platform_only`. An intentionally absent upstream or previous counterpart uses
`{"absent": true, "reason": "Concrete absence explanation"}`. Do not invent a symbol
for platform-only additions or removed paths. The tool verifies mapped identities;
the agent still reviews whether the mapping and MACA rationale are correct. `sha256`
identifies the entire referenced source file; `symbol` must exist in it (`<module>`
can represent the whole module). References default to the selected vLLM checkout;
`previous.source` may identify a different prior-source checkout when needed. Reviewable
target-vs-local diffs and mapping records are retained under the run's `review/` directory.

Persist semantic producer/consumer dependencies separately. `contract-put --file contract.json`
accepts:

```json
{
  "id": "cache-layout",
  "upstream": "TARGET_REVISION:path:Symbol",
  "summary": "Producer and consumer share the physical cache layout",
  "invariants": ["Specify concrete dtype, axes, strides and index units"],
  "producer": "cache-task",
  "consumers": ["attention-layout"]
}
```

Register the support matrix with `matrix-put --file matrix.json`. This replaces the
whole matrix, so retain all intended rows when updating it:

```json
{
  "rows": [{
    "id": "bf16-single-gpu",
    "status": "supported",
    "configuration": {"dtype": "bfloat16", "tensor_parallel_size": 1},
    "requires": ["component_numeric", "e2e"],
    "tasks": ["attention-layout"]
  }]
}
```

Each row declares its exact configuration, required evidence kinds and linked task
IDs. The task's `--configuration` is a label; row membership comes from `row.tasks`,
while `row.configuration` is the expected actual dictionary reported by the child.
Do not bind one task to rows with conflicting configuration dictionaries.
Full runs need at least one supported row and current `e2e` plus every other required
kind for **each supported row**. An `unsupported` row needs `rejection_reason` and
an actual contract/production rejection result. A label or exclusion note cannot
stand in for the rejection check. Component numerical and production checks also
need matrix binding in component mode; a component run makes only that scoped claim.

Before numerical checks, register the frozen contract with
`numeric-put --id TASK --file numerical.json`:

```json
{
  "reference": {"description": "Independent reference implementation", "revision": "REFERENCE_REVISION"},
  "seed": 0,
  "tolerances": {"atol": 0.001, "rtol": 0.001},
  "nan_policy": "forbid",
  "inf_policy": "forbid",
  "cases": ["prefill-short", "decode-multi-token"]
}
```

Changing an existing numerical contract requires `--note TEXT` explaining the
technical basis. The previous contract remains in `acceptance_history` and affected
evidence becomes stale. Do not loosen tolerances solely to make a failing test pass.

The reference must be independent of the kernel under test. NaN/Inf policies are
`forbid` or `match_reference`; use the documented child-result matching fields when
special values are allowed: the relevant case must report `nan_matches_reference: true`
or `inf_matches_reference: true` after comparing actual patterns and Inf signs. Changing mappings, contracts or matrix obligations makes
affected evidence stale. It does not turn older results into evidence for the new plan.

The agent copies the exact target upstream module/function before reapplying verified
MACA semantics. Preserve upstream structure and semantic roles; annotate necessary
differences concisely with `NOTE(metax):`. Do not optimize during adaptation.

Useful commands, all taking `--run-dir`:

| Command | Purpose |
| --- | --- |
| `scope-set --scope TEXT` | Update the confirmed run scope without reinitializing; records history and invalidates task evidence. |
| `task-set --id ID --status STATUS --note TEXT --next-action TEXT` | Save progress/blockers; done requires current evidence and done dependencies. |
| `run-check --id ID --kind KIND --configuration CONFIG --input-file PATH --timeout SECONDS -- COMMAND ARGS` | Run a fresh bounded subprocess and retain input hashes, output, provenance and case results; repeat `--input-file` for indirect inputs. |
| `checkpoint --note TEXT --next-action TEXT` | Save a handoff and invalidate evidence affected by source changes. |
| `status` | Reconcile source hashes and regenerate the visible TODO. |
| `resume` | Check environment/source identity, recover interrupted work and clean recorded checks. |
| `refresh` | Collect a new full environment snapshot; changed identity revokes approval. |
| `matrix-put --file PATH` | Replace the structured support matrix and invalidate affected evidence. |
| `mapping-put --id ID --file PATH` | Register source mappings and MACA dispositions for every task file. |
| `numeric-put --id ID --file PATH [--note TEXT]` | Freeze numerical acceptance; changing an existing contract needs a technical reason. |
| `migrate --note TEXT` | Back up an old schema and migrate it with prior evidence marked stale. |
| `accept-policy-change --note TEXT` | Record reviewed tool/rule changes before new checks; old evidence becomes stale. |
| `finish` | Check current acceptance; full mode requires inventory coverage and every supported matrix row's e2e/other obligations. |

Evidence kinds: `static`, `contract`, `component_numeric`, `closure`, `weight_loading`,
`e2e`, `distributed`, `graph`, `review`. The command must actually assert the expected
behavior; exit zero by itself cannot establish adequate coverage. A later failure
supersedes an earlier pass. Contract/source/environment changes invalidate evidence.
Declare scripts, reference implementations, fixtures and indirect data/config inputs
using repeatable `--input-file`; their content hashes are checked again before evidence
can count. Direct command scripts and PATH executables are hashed too, including
extensionless scripts; retargeting a declared symlink invalidates its old evidence.
Each check saves `acceptance.json` with the matrix, mapping, numerical contract and
tool/rule identity used for that invocation. Evidence records its policy fingerprint.

Production kinds require actual child module origins and selected dispatch in JSON
written to `METAX_UPGRADE_ORIGINS_FILE`. `component_numeric` requires actual installed
callable/extension origins. Both require the actual structured configuration. Numerical
results are required for a registered numerical contract; `component_numeric` always
requires such a contract. The runner sets
`METAX_UPGRADE_NUMERICAL_CONTRACT_FILE` to the persisted contract and
`METAX_UPGRADE_NUMERICAL_CONTRACT_DIGEST` to its digest. Read these in the validating
child; do not supply an invented digest or prewrite its report in the coordinator.

Numerical results contain `numerical.contract_digest` and a `cases` array. Every
registered case must report `id`, `max_abs`, `reference_scale`, `nan_count`, `inf_count`
and `passed: true`. The tool independently checks
`max_abs <= atol + rtol * reference_scale` and the frozen NaN/Inf policy; `passed` alone
cannot approve a result. Report measured error and reference scale from the independent
comparison, not constants selected to satisfy the formula. The following production
report illustrates a task with a numerical contract; all paths, dispatch and numeric
values must come from the actual validating child:

```json
{
  "vllm": "/actual/runtime/vllm/__init__.py",
  "vllm_metax": "/actual/runtime/vllm_metax/__init__.py",
  "dispatch": {"backend": "actual.module.SelectedBackend"},
  "configuration": {"dtype": "bfloat16", "tensor_parallel_size": 1},
  "numerical": {
    "contract_digest": "VALUE_OF_METAX_UPGRADE_NUMERICAL_CONTRACT_DIGEST",
    "cases": [
      {"id": "prefill-short", "max_abs": 0.0002, "reference_scale": 1.0, "nan_count": 0, "inf_count": 0, "passed": true},
      {"id": "decode-multi-token", "max_abs": 0.0003, "reference_scale": 1.1, "nan_count": 0, "inf_count": 0, "passed": true}
    ]
  }
}
```

A component numerical report uses `callables` with the actual callable/extension
identities instead of requiring MetaX package imports solely for provenance, and
retains the same configuration/numerical fields. Every declared configuration field
must match; the child may report additional observed fields. A deliberate unsupported-row check
writes `rejected: true`, a nonempty `rejection_reason` and the matching actual
`configuration` to the same child report file. It must assert the actual expected
capability rejection; simply writing an expected message is insufficient.

Review commands can check a persisted disposition artifact; also record the review
judgment and limitations in task notes. Arbitrary commands use argument vectors,
without an implicit shell. Only production kinds retain the normal plugin configuration.

Initialization persists its recovery phase before archiving the initial dirty worktree
in `baseline/metax-source.zip` with its manifest. The archive is built atomically;
resume an interrupted baseline phase before adapting code, rather than replacing the
original snapshot with a later edited tree.
`state.json` is authoritative and atomically replaced; `TODO.md` is a recoverable view.
Runs under `.metax-upgrade/` are ignored by Git but remain visible in the workspace.
Keep them on durable storage when moving machines or handing work to another AI.
State uses a Linux advisory lock. Checks run under a supervisor that kills their process
group if the coordinator dies or their timeout expires; a launch handshake records
process birth identity before starting the workload. Programs that deliberately detach
into another session must manage and record their own cleanup.

`--mode component` limits completion to declared component obligations. It makes no
full-repository claim. Full mode requires every inventoried production/build/requirements
file to have a disposition task and a mapping. Each supported matrix row must pass
normal-entry e2e and every other declared evidence kind. The agent must still review
whether the matrix covers the intended model/configuration scope and whether numerical
references, case selection and measured reports are semantically valid.

Older run schemas must use `migrate --note TEXT`; the command preserves the original
state backup and retains old evidence as stale. Fill missing mappings/matrix/numerical
contracts and rerun required checks. When current tool or shared-rule identity changes,
review the difference and use `accept-policy-change --note TEXT` before new checks.
This records a workflow-version acknowledgment and invalidates old evidence; it is
separate from user confirmation of the environment. Freshly unchanged environment
facts retain their actual prior approval; changed facts still need confirmation.

Offline helper validation (independent of the repository test fixtures):

```bash
python -m unittest discover -s tools/metax_upgrade/tests -p 'test_*.py' -v
ruff check tools/metax_upgrade
ruff format --check tools/metax_upgrade
```
