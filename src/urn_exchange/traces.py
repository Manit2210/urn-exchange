"""Export match records as a shareable trace file (``.jsonl.gz``), without opaque reasoning blobs.

A trace is the runner's match record (one JSON object per line; see docs/harness.md "Data and trace
format"): seating, every step (view, action, error), the result with per-seat metrics and the revealed
deal, and per seat the system prompt and every model call (reply text, tool calls, usage, cost, latency,
stop reason, upstream host and readable reasoning). Only the last record per match id is kept, and only
matches that finished ``ok``.

Removed: provider reasoning state that is opaque and only meaningful to the provider that issued it,
namely encrypted reasoning (OpenRouter ``reasoning_details`` of type ``reasoning.encrypted``, OpenAI
``encrypted_content``), signatures on thinking blocks, and Gemini ``thought_signature`` blobs. Each removal
leaves a marker (``data_removed``, ``signature_removed``, ``thought_signature_removed``) so the shape of
the reasoning is still visible. Readable thinking text and summaries are kept.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any, Iterable

# Keys of opaque blobs anywhere inside provider_state.
_BLOB_KEYS = ("signature", "thought_signature", "encrypted_content")


def _strip_detail(d: Any) -> Any:
    if not isinstance(d, dict):
        return d
    d = dict(d)
    if d.get("type") in ("reasoning.encrypted", "redacted_thinking") and d.get("data") is not None:
        d["data"] = None
        d["data_removed"] = True
    for k in _BLOB_KEYS:
        if k in d:
            d.pop(k)
            d[f"{k}_removed"] = True
    return d


def strip_provider_state(state: dict) -> dict:
    out: dict[str, Any] = {}
    for key, val in (state or {}).items():
        if key == "openrouter_reasoning_details" and isinstance(val, list):
            out[key] = [_strip_detail(d) for d in val]
        elif key == "gemini_content" and isinstance(val, dict):
            out[key] = {**val, "parts": [_strip_detail(p) for p in val.get("parts") or []]}
        elif key == "anthropic_content" and isinstance(val, list):
            out[key] = [_strip_detail(b) for b in val]
        elif key == "openai_reasoning_items" and isinstance(val, list):
            out[key] = [_strip_detail(i) for i in val]
        else:
            out[key] = val
    return out


def strip_record(rec: dict, dataset: str | None = None) -> dict:
    """A copy of a match record with opaque reasoning blobs removed (see the module docstring)."""
    rec = json.loads(json.dumps(rec))
    for p in rec.get("players") or []:
        for c in p.get("calls") or []:
            if c.get("provider_state"):
                c["provider_state"] = strip_provider_state(c["provider_state"])
    if dataset is not None:
        rec = {"dataset": dataset, **rec}
    return rec


def final_ok_records(records: Iterable[dict]) -> list[dict]:
    last: dict[str, dict] = {}
    for r in records:
        if r.get("record") == "match":
            last[r["match_id"]] = r
    return [r for r in last.values() if r.get("status") == "ok"]


def long_tokens(obj: Any, min_len: int = 400) -> list[str]:
    """Whitespace-free strings of at least ``min_len`` characters (left-over blobs), for checks."""
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and len(x) >= min_len and not any(ch.isspace() for ch in x):
            out.append(x)

    walk(obj)
    return out


def export(sources: list[tuple[str | Path, str]], out_path: str | Path) -> dict:
    """Write ``out_path`` (gzip JSONL) from ``[(run dir or matches.jsonl, dataset label), ...]``."""
    from urn_exchange.runner import load_records

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    blobs = 0
    # mtime=0 and no file name in the gzip header: the same records always give the same bytes (checksum).
    with open(out, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8") as f:
        for src, label in sources:
            recs = sorted(final_ok_records(load_records(src)), key=lambda r: r["match_id"])
            for r in recs:
                s = strip_record(r, label)
                blobs += len(long_tokens(s))
                f.write(json.dumps(s, sort_keys=True, ensure_ascii=False) + "\n")
            counts[label] = len(recs)
    return {"path": str(out), "bytes": out.stat().st_size, "records": counts, "long_tokens_left": blobs}
