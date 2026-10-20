# Shared validation and defect review

Read this for compatibility reviews and adaptations before selecting tests or making
correctness claims. Domain skills supply the concrete cases; persistence and CLI
provenance formats live in [run-state.md](run-state.md).

## Validate while dependencies are unfinished

Begin with self-contained isolated scripts written from the fixed target contracts.
Do not trust repository tests, fixtures/conftest or old `/tmp` files before reviewing
their dependencies. Keep reproducible scripts and raw logs in the durable run directory.
Use repository integration/regression tests once the relevant fixtures and real
production dependency closures are ready.

Run candidates in fresh subprocesses with the selected interpreter, actual runtime
cwd and explicit process-local import plan. Record actual package/plugin, callable
and extension origins from the validating child; a metadata version or parent-process
resolution is insufficient. Candidate execution discovers failures without making
unrelated MetaX modules trusted. Recheck source/runtime correspondence after changes
to dependencies, checkouts, interpreter, cwd or import plan.

Stubs, file-based loading, optional-dependency bypasses and `--confcutdir` can isolate
a contract. Record exactly what they bypass, retain the real function under review,
and rerun the production path when ready. A bypassed initializer or mocked dependency
does not establish startup, production reachability, installed-kernel support or e2e
correctness. Do not silently alter the environment or remove required coverage to
turn a blocked check into a pass.

## Match evidence to the claim

Select acceptance obligations per configuration before editing. Evidence is a set of
independent obligations, not a single ascending score:

| Evidence | What it establishes |
| --- | --- |
| Static | Source/signature/AST/contract reasoning against the fixed target. |
| Isolated contract | The asserted behavior within recorded stubs or bypass boundaries. |
| Component numerical | The real installed component agrees with an independent reference for specified cases. |
| Real closure | Actual adapter startup, registration, dispatch, wrappers and asserted behavior for the exercised dependency/configuration closure. |
| Weight loading / e2e | Normal-entry construction, real checkpoint loading and execution for the declared cases. |
| Distributed / graph | The exercised multi-process/multi-GPU or capture/replay behavior. |
| Performance | Measurements under a recorded benchmark configuration. |

Use actual installed kernels for numerical claims. The reference must not reuse the
same potentially faulty kernel. Imports/signatures cannot prove numerical behavior;
a raw kernel check cannot prove wrapper dispatch; a dummy/trimmed model proves only
the paths it preserves, not full-model accuracy or real-weight loading. Upstream flags
and support in individual dimensions do not establish their joint MetaX configuration.

Before numerical execution, persist the numerical contract: independent reference and
its version/source, fixed seeds, absolute/relative tolerance, NaN/Inf policy and required
cases. Declare scripts, fixtures, reference files and other test inputs as hashed check
inputs. The validating child must report its actual configuration, contract digest,
per-case results and error metrics; its measured values must meet the frozen policy.
A command's success or a nonempty provenance object cannot substitute for these results.
Changing the contract or any recorded input invalidates affected evidence.
An existing numerical contract can change only with a recorded technical reason;
retain its previous version. Do not relax tolerances solely to pass a failing case.

Choose the least expensive checks that expose the changed behavior, including boundary
and rejected cases. Run relevant syntax/format/lint/type/diff checks. For comment or
docstring-only code changes, an AST comparison excluding docstrings can verify that
executable code is unchanged without requiring a model suite. Once appropriate checks
pass, expand only for new changes, failures or unresolved concerns. Numerical success
does not establish speed, distributed support or graph correctness without those checks.

Record task/contract IDs, exact command/inputs/configuration, expected and observed
results, source/environment fingerprints, actual runtime origins, component/binary
identity, bypasses and limitations. Separate checks run, checks passed and obligations
still pending; keep failed and superseded evidence visible. Missing hardware, checkpoints
or components remain explicit unverified/blocked obligations.

## Attribute failures and re-review

For each suspected defect, inspect the equivalent path in the exact target upstream
revision, including dispatch and relevant dependencies. Where feasible run the same
discriminating reproduction. Classify adaptation regression, existing MetaX defect,
shared upstream defect, installed component/ABI issue, environment mismatch or blocked
prerequisite. Distinguish source-based suspicion from a reproduced upstream failure
and limit upstream claims to the revision actually inspected. Specialist-specific
repair boundaries still apply; attribution alone does not authorize broader changes.

For a suspected installed-component defect, isolate a direct call with valid inputs
at that component's boundary. Distinguish its result from wrapper transformations
before attributing the failure to the component or adapter.

Verify behavioral claims in comments against implementation and executable checks.
After edits, revisit every prior finding as fixed, partially fixed, still present or
masked by an earlier failure/capability rejection. A first exception can hide later
bugs; do not stop at fixing it. Recheck for concurrent source changes before final
reporting and invalidate stale evidence for changed units and affected consumers.
