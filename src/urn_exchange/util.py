"""Shared helpers: canonical JSON, hashing, a small JSON-schema checker and secret redaction."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
from typing import Any, Iterable

# --------------------------------------------------------------------------- canonical JSON


def _normalise(obj: Any) -> Any:
    """Convert tuples to lists; dict keys become strings."""
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer() and abs(obj) < 2**53:
        # 1.0 and 1 hash the same; JSON cannot distinguish them for providers anyway.
        return int(obj)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- JSON schema (subset)

_TYPES = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate a subset of JSON schema (type, required, properties, items, enum,
    additionalProperties=false, minimum/maximum). Returns a list of error strings."""
    errors: list[str] = []
    typ = schema.get("type")
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        ok = False
        for t in types:
            py = _TYPES.get(t)
            if py is None:
                ok = True
                break
            if isinstance(value, bool) and t in ("number", "integer"):
                continue
            if isinstance(value, py):
                ok = True
                break
        if not ok:
            return [f"{path}: expected {typ}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required '{req}'")
        for k, v in value.items():
            if k in props:
                errors.extend(validate_schema(v, props[k], f"{path}.{k}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property '{k}'")
    if isinstance(value, (list, tuple)) and "items" in schema:
        for i, v in enumerate(value):
            errors.extend(validate_schema(v, schema["items"], f"{path}[{i}]"))
    return errors


# --------------------------------------------------------------------------- secrets

_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()
REDACTED = "[REDACTED]"


def register_secret(value: str) -> None:
    if value and len(value) >= 6:
        with _SECRETS_LOCK:
            _SECRETS.add(value)


def redact(text: str, extra: Iterable[str] = ()) -> str:
    """Replace every registered secret (API keys read by this process) in ``text``."""
    if not isinstance(text, str):
        text = str(text)
    with _SECRETS_LOCK:
        secrets = set(_SECRETS)
    secrets.update(s for s in extra if s)
    for s in sorted(secrets, key=len, reverse=True):
        text = text.replace(s, REDACTED)
    return text


class MissingCredentialError(RuntimeError):
    pass


class ApiKey:
    """An API key read from an environment variable. The value never appears in repr/str."""

    __slots__ = ("env_var", "_value")

    def __init__(self, env_var: str, value: str | None = None):
        self.env_var = env_var
        self._value = value
        if value:
            register_secret(value)

    @classmethod
    def from_env(cls, env_var: str, required: bool = True) -> "ApiKey":
        val = os.environ.get(env_var)
        if required and not val:
            raise MissingCredentialError(f"environment variable {env_var} is not set")
        return cls(env_var, val)

    @property
    def present(self) -> bool:
        return bool(self._value)

    def reveal(self) -> str:
        """Only for handing to an SDK client constructor."""
        if not self._value:
            raise MissingCredentialError(f"environment variable {self.env_var} is not set")
        return self._value

    def __repr__(self) -> str:
        return f"ApiKey(env_var={self.env_var!r}, value={REDACTED if self._value else None})"

    __str__ = __repr__

    def __reduce__(self):  # never pickle the value
        raise TypeError("ApiKey is not picklable")
