# Contract-driven orchestration

Read this when planning a full upgrade or expanding its dependency graph. Apply
common's [validation policy](../../vllm-metax-upgrade-common/references/validation.md)
and [run-state mechanics](../../vllm-metax-upgrade-common/references/run-state.md)
for evidence and persistence; this reference defines coordinator-specific work.

## Build the graph

The graph contains adaptation units and installed component boundaries. Assign each
unit its files/symbols, upstream counterpart, incoming/outgoing contracts, configuration
conditions, required evidence, owner and next action. Track at least these edge types:

| Edge | Failure that it can explain |
| --- | --- |
| Import | Parent initializer, decorator or base-class definition fails. |
| Startup | Platform/entry-point ordering activates unfinished work. |
| Side effect | Patch, registry mutation or early imported alias binds another object. |
| Call | Parameter, return, MRO or inherited-state contract changed. |
| Data | Cache/index/weight layout or scale semantics disagree. |
| Conditional | Quantization, phase, parallel, MTP or graph branch selects another path. |

Static scanning seeds the graph; source review and runtime tracing refine it. Find
strongly connected components before ordering work. Adapt cyclic interfaces together,
then execute their combined startup closure. When eager initializers pull unrelated
modules into that closure, either schedule the closure or separately justify an
initialization change preserving registration order. A test-only initializer bypass
does not resolve the production startup edge.

## Assign and schedule units

Dispatch model, attention, registry and patch units to their specialists. Apply common
with explicit contract review for platform, distributed, worker, allocator, foundational
ops and build/ABI units without a specialist. Keep each worker on an assigned unit;
return discoveries, dispositions and evidence to the authoritative run.

Enumerate transitive consumer tasks before a shared contract changes. A blocked
integration obligation need not stop independent units. When a producer becomes ready,
schedule eligible consumers; when it changes, propagate evidence invalidation and
revalidation through the graph rather than treating domain completion as permanent.

## Expand and close the support matrix

Complete a minimal normal-entry vertical path early, then expand model variant,
dtype/quantization, prefill/decode, checkpoint loading, TP/PP/EP/DCP, MTP, graph and
optional-feature rows as agreed. Persist each row's structured configuration, linked
tasks and acceptance vector; record deliberate rejection reasons on unsupported rows.
Track actual production reachability and configuration-matched numerical results,
not only per-component commands or prose support claims.

Full mode assigns every baseline production/build/requirements file to a disposition
task. At least one row must be supported. Every supported row declares and passes its
own normal-entry `e2e` obligation
and all additional required evidence kinds. The child reports the actual configuration
and frozen numerical-contract results so evidence from another row cannot satisfy it.
Required blocked
checks keep completion pending. Deliberately unsupported combinations need intended
capability rejection and must be visible in the matrix. Hardware, checkpoint or
component prerequisites belong in the handoff with their recovery conditions.
