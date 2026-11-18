"""Response cache: one JSON file per request hash under ``cache_dir``.

The runner's cache key includes the run id, match, seat and every message so far, so a read can only
replay a call this run already paid for (a resumed match picks up where it stopped); different runs,
seats or seeds never share an entry.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from urn_exchange.providers.base import ProviderResponse


class ResponseCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, request_hash: str) -> Path:
        return self.root / request_hash[:2] / f"{request_hash}.json"

    def get(self, request_hash: str) -> ProviderResponse | None:
        p = self._path(request_hash)
        if not p.exists():
            return None
        return ProviderResponse.from_dict(json.loads(p.read_text()))

    def put(self, request_hash: str, response: ProviderResponse) -> None:
        p = self._path(request_hash)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(response.to_dict(), sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, p)

    def __contains__(self, request_hash: str) -> bool:
        return self._path(request_hash).exists()

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json"))
