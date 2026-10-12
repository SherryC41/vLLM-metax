# Runtime patches

Runtime patches provide temporary vLLM fixes or MetaX compatibility changes when
an existing registry or extension point cannot be used.

> Patches depend on vLLM implementation details. Keep them small, tested, and
> easy to remove after the upstream issue is fixed.

The current upstream compatibility review and removal decisions are recorded in
[AUDIT.md](AUDIT.md).

## Quick start

1. Choose a template:
   - `template/module_attr.py` for module functions, classes, or Triton kernels.
   - `template/class_method.py` for instance methods.
2. Copy the template into `bugfix/`, `enhancement/`, or `performance/`.
3. Complete the header and replace every `TODO`.
4. Import the new patch module from the nearest `__init__.py`.
5. Add a focused regression test.

Importing a patch module applies it immediately. Module-attribute patches also
update same-name exports in already-loaded `vllm.*` modules when they reference
the original object. Future imports resolve the replacement normally. Other
objects with the same name and captured originals in MetaX modules are preserved.
Renamed exports (`from ... import target as other_name`) are not scanned.

Explicitly patching the same target twice raises `RuntimeError`. Stacked
decorators may still name an automatically updated alias: the first explicit
installation preserves its original metadata, and subsequent explicit
installations fail. Class-method patches preserve descriptors on the shared
class; they do not scan module exports for bound methods or property getters.

## Installation flow

The decorator applies the replacement when the patch module is imported. A name
such as `target_function` selects the module path; `TargetClass.forward` selects
the class path. This diagram is also kept in the `utils.py` module docstring:

```text
                 +--------------------------------------------+
                 |   patch(module, attribute)(replacement)    |
                 +--------------------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |    attribute contains a dot?     |
                      +----------------------------------+
                                       |
                 +---------------------+---------------------+
                 | no                                        | yes
                 v                                           v
+----------------------------------+        +----------------------------------+
|         MODULE ATTRIBUTE         |        |         CLASS ATTRIBUTE          |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|       Resolve module/name        |        |       Resolve owner class        |
|      (infer name if absent)      |        |        and attribute name        |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|      Read original (static)      |        |     Read descriptor (static)     |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|      Check target state [A]      |        |      Reject duplicate patch      |
|      (may return or raise)       |        |                                  |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|         Record original          |        |        Rebuild descriptor        |
|        Set module export         |        |   method/static/class/property   |
|                                  |        |    (preserve setter/deleter)     |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|       If original existed:       |        |         Record original          |
|    sync loaded vllm.* aliases    |        |       Set class attribute        |
|  by same name + object identity  |        |                                  |
+----------------------------------+        +----------------------------------+
                 |                                           |
                 v                                           v
+----------------------------------+        +----------------------------------+
|   Return replacement unchanged   |        | Return installed class attribute |
|    (keep JIT/autotune object)    |        |                                  |
+----------------------------------+        +----------------------------------+
```

Missing targets raise `AttributeError` unless `allow_missing=True`. On the class
path, duplicate detection checks the underlying implementation, and installation
preserves the existing descriptor and any property setter/deleter.

The module path has one exception to duplicate rejection: a stacked decorator
may explicitly name an alias that this installation already updated
automatically. That alias is promoted once, retaining its recorded original:

```text
Module target state [A]:

                  +------------------------------------------+
                  |  Same replacement AND automatic alias?   |
                  +------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 | yes                                       | no
                 v                                           v
+----------------------------------+        +----------------------------------+
|    Promote alias to explicit     |        |      Target is replacement       |
|      Keep recorded original      |        |       OR has patch marker?       |
|     Remove automatic marker      |        |                                  |
|        Return immediately        |        |                                  |
+----------------------------------+        +----------------------------------+
                                                             |
                                                   +---------+---------+
                                                   | yes               | no
                                                   v                   v
                                            +--------------+    +--------------+
                                            | RuntimeError |    |   Continue   |
                                            |              |    | installation |
                                            +--------------+    +--------------+
```

For example, if two modules export the same original object `O`, patching its
defining export to `R` updates both bindings and records both originals:

```text
+------------------------------+    +------------------------------------------+
|            Before            |    |     After patch(source, "target")(R)     |
|                              |    |                                          |
|      source.target -> O      |    |            source.target -> R            |
|   vllm.caller.target -> O    |    |         vllm.caller.target -> R          |
| MetaX captured original -> O |    |       MetaX captured original -> O       |
+------------------------------+    +------------------------------------------+

                                    +------------------------------------------+
                                    |          R.__vllm_metax_patch__          |
                                    |                                          |
                                    |            source.target -> O            |
                                    |         vllm.caller.target -> O          |
                                    +------------------------------------------+
```

The scan only visits loaded `vllm.*` modules, compares the same attribute name
and object identity, and records alias updates directly without recursively
calling `patch()`. Future imports obtain `R` from the defining export. Renamed
exports and references inside local variables or containers are outside this
scan. Class aliases share their owner class, so patching a class attribute
updates that owner without scanning module exports.

## Common examples

### Module function

The target name is inferred from the replacement function name:

```python
from vllm_metax.patch.utils import patch


@patch("vllm.some_module")
def target_function(arg: int) -> int:
    # Copy the complete upstream implementation and mark the MetaX change.
    ...
```

Specify the attribute when the names differ:

```python
@patch("vllm.some_module", "target_function")
def metax_target_function(arg: int) -> int:
    ...
```

### Class method

Use a dotted class attribute path. The target class is resolved internally and
does not need to be imported by the patch file:

```python
@patch("vllm.some_module", "TargetClass.forward")
def forward(self, hidden_states):
    ...
```

Keep `@patch` outside method descriptors:

```python
@patch("vllm.some_module", "TargetClass.from_config")
@classmethod
def from_config(cls, config):
    ...
```

The same form supports `staticmethod` and `property`.

### Triton kernel

`@patch` must be the outermost decorator so the final Triton object is installed:

```python
@patch("vllm.some_module", "target_kernel")
@triton.heuristics({...})
@triton.autotune(configs=[...], key=[...])
@triton.jit
def target_kernel(...):
    ...
```

## Patch layout

- `bugfix/`: upstream and integration fixes.
- `enhancement/`: temporary compatibility features.
- `performance/`: performance replacements.
- `template/`: ready-to-copy patch examples.
- `utils.py`: the shared `@patch` implementation.

## Keep in mind

- Prefer `vllm_metax/registry/`, attention backends, or platform configuration
  hooks when they can implement the change.
- Preserve the upstream name, signature, decorators, and unchanged code so future
  version updates are easy to diff.
- Mark only changed lines with the `MetaX Modification` comments from the
  templates.
- Use `allow_missing=True` only when intentionally adding a new compatibility
  attribute; normal patches require the target to exist.
- Record affected versions and a concrete removal condition in every patch.
