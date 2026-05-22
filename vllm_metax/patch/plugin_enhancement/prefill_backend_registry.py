# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from collections.abc import Callable
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum


def get_path(self) -> str:
    """Get the class path for this backend (respects overrides).

    Returns:
        The fully qualified class path string

    Raises:
        ValueError: If Backend.CUSTOM is used without being registered
    """
    path = _MLA_PREFILL_OVERRIDES.get(self, self.value)
    if not path:
        raise ValueError(
            f"MLA prefill backend {self.name} must be registered before "
            f"use. Use register_mla_prefill_backend("
            f"MLAPrefillBackendEnum.{self.name}, "
            f"'your.module.YourClass')"
        )
    return path


def is_overridden(self) -> bool:
    """Check if this backend has been overridden."""
    return self in _MLA_PREFILL_OVERRIDES


def clear_override(self) -> None:
    """Clear any override for this backend, reverting to the default."""
    _MLA_PREFILL_OVERRIDES.pop(self, None)


_MLA_PREFILL_OVERRIDES: dict[MLAPrefillBackendEnum, str] = {}


def register_mla_prefill_backend(
    backend: MLAPrefillBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override an MLA prefill backend implementation.

    Args:
        backend: The MLAPrefillBackendEnum member to register.
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op.

    Examples:
        # Override an existing MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn(MLAPrefillBackend):
            ...

        # Register a custom third-party MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
        class MyCustomPrefillBackend(MLAPrefillBackend):
            ...

        # Direct registration
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "my.module.MyCustomPrefillBackend"
        )
    """

    def decorator(cls: type) -> type:
        _MLA_PREFILL_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _MLA_PREFILL_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator


MLAPrefillBackendEnum.get_path = get_path
MLAPrefillBackendEnum.is_overridden = is_overridden
