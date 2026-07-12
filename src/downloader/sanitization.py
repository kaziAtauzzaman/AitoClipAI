"""Deterministic JSON-safe sanitization for provider metadata."""

from base64 import b64encode
from collections.abc import Mapping, Sequence, Set
from datetime import date, datetime, time
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Protocol, TypeAlias


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class MetadataSanitizer(Protocol):
    """Convert arbitrary provider metadata into deterministic JSON values."""

    def sanitize(self, value: object) -> JsonValue:
        """Return a JSON-safe value without mutating the input."""


class JsonMetadataSanitizer:
    """Recursively sanitize metadata while preserving useful value types."""

    _TYPE_MARKER = "__aitoclipai_unsupported_type__"
    _CYCLE_MARKER = "__aitoclipai_cycle__"
    _BYTES_MARKER = "__aitoclipai_bytes_base64__"
    _FLOAT_MARKER = "__aitoclipai_non_finite_float__"

    def sanitize(self, value: object) -> JsonValue:
        """Return a deterministic JSON-safe copy of ``value``."""

        return self._sanitize(value, active_ids=set())

    def _sanitize(self, value: object, active_ids: set[int]) -> JsonValue:
        if value is None or isinstance(value, bool | int | str):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return {self._FLOAT_MARKER: _non_finite_float_name(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime | date | time):
            return value.isoformat()
        if isinstance(value, Enum):
            return self._sanitize(value.value, active_ids)
        if isinstance(value, bytes | bytearray | memoryview):
            return {self._BYTES_MARKER: b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, Mapping):
            return self._sanitize_mapping(value, active_ids)
        if isinstance(value, Set) and not isinstance(value, str | bytes):
            return self._sanitize_set(value, active_ids)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return self._sanitize_sequence(value, active_ids)
        return {self._TYPE_MARKER: _qualified_type_name(value)}

    def _sanitize_mapping(
        self,
        value: Mapping[object, object],
        active_ids: set[int],
    ) -> JsonValue:
        identifier = id(value)
        if identifier in active_ids:
            return {self._CYCLE_MARKER: True}
        active_ids.add(identifier)
        try:
            return {
                _json_key(key): self._sanitize(item, active_ids)
                for key, item in value.items()
            }
        finally:
            active_ids.remove(identifier)

    def _sanitize_sequence(
        self,
        value: Sequence[object],
        active_ids: set[int],
    ) -> JsonValue:
        identifier = id(value)
        if identifier in active_ids:
            return {self._CYCLE_MARKER: True}
        active_ids.add(identifier)
        try:
            return [self._sanitize(item, active_ids) for item in value]
        finally:
            active_ids.remove(identifier)

    def _sanitize_set(
        self,
        value: Set[object],
        active_ids: set[int],
    ) -> JsonValue:
        identifier = id(value)
        if identifier in active_ids:
            return {self._CYCLE_MARKER: True}
        active_ids.add(identifier)
        try:
            sanitized = [self._sanitize(item, active_ids) for item in value]
            return sorted(sanitized, key=_canonical_json)
        finally:
            active_ids.remove(identifier)


def _json_key(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, bool | int | float):
        return str(value)
    return f"<{_qualified_type_name(value)}>"


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _non_finite_float_name(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "infinity" if value > 0 else "-infinity"
