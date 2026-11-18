"""Spend ledger and budget guard.

The ledger is a small SQLite file (``ledger`` in the config, default ``results/spend.sqlite``) shared by
every thread and process of a run. Before each paid call the guard atomically (``BEGIN IMMEDIATE``)
checks ``spent by this run + projected worst case <= budget_usd`` and inserts a reservation for the
projected cost; after the call the reservation is replaced by the billed cost. A crash leaves the
reservation in place, which keeps the ledger conservative. The budget is scoped to the run id, so a
resumed run keeps counting what it already spent.

The guard is a safety net on top of, not instead of, a spend limit on your provider account: set one
there too.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from urn_exchange.util import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode_id TEXT,
    request_hash TEXT,
    model TEXT,
    kind TEXT NOT NULL,
    amount_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_run ON spend(run_id);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def connect(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=60000")
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if immediate:
                    conn.execute("ROLLBACK")
                raise
            else:
                if immediate:
                    conn.execute("COMMIT")
        finally:
            conn.close()

    def run_spend(self, run_id: str, conn: sqlite3.Connection | None = None) -> float:
        q = "SELECT COALESCE(SUM(amount_usd),0) FROM spend WHERE run_id=?"
        if conn is not None:
            return float(conn.execute(q, (run_id,)).fetchone()[0])
        with self.connect() as c:
            return self.run_spend(run_id, c)

    def total_spend(self) -> float:
        with self.connect() as c:
            return float(c.execute("SELECT COALESCE(SUM(amount_usd),0) FROM spend").fetchone()[0])


class SpendCapExceeded(RuntimeError):
    error_class = "spend_cap"


@dataclass
class Reservation:
    id: int | None
    projected_usd: float


class SpendGuard:
    def __init__(self, ledger: Ledger, budget_usd: float, *, run_id: str, dry_run: bool = False):
        if budget_usd is None or budget_usd < 0:
            raise ValueError("budget_usd must be a non-negative number")
        self.ledger = ledger
        self.budget_usd = float(budget_usd)
        self.run_id = run_id
        self.dry_run = dry_run
        self.dry_run_projected = 0.0
        self.dry_run_would_exceed = False

    def spent(self) -> float:
        return self.ledger.run_spend(self.run_id)

    def reserve(self, projected_usd: float, *, episode_id: str, request_hash: str, model: str) -> Reservation:
        projected_usd = max(0.0, float(projected_usd))
        if self.dry_run:
            self.dry_run_projected += projected_usd
            self.dry_run_would_exceed = self.dry_run_projected > self.budget_usd
            return Reservation(None, projected_usd)
        with self.ledger.connect(immediate=True) as c:
            spent = self.ledger.run_spend(self.run_id, c)
            if spent + projected_usd > self.budget_usd:
                raise SpendCapExceeded(
                    f"refusing call: run {self.run_id} spent {spent:.4f} + projected worst case {projected_usd:.4f} "
                    f"> budget_usd {self.budget_usd:.2f}")
            cur = c.execute(
                "INSERT INTO spend(run_id, episode_id, request_hash, model, kind, amount_usd, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (self.run_id, episode_id, request_hash, model, "reservation", projected_usd, now_iso()),
            )
            return Reservation(cur.lastrowid, projected_usd)

    def settle(self, res: Reservation, actual_usd: float) -> None:
        if res.id is None:
            return
        with self.ledger.connect(immediate=True) as c:
            c.execute("UPDATE spend SET kind='actual', amount_usd=? WHERE id=?", (max(0.0, float(actual_usd)), res.id))

    def release(self, res: Reservation) -> None:
        """The call was rejected before the provider did any work (not billable)."""
        if res.id is None:
            return
        with self.ledger.connect(immediate=True) as c:
            c.execute("DELETE FROM spend WHERE id=? AND kind='reservation'", (res.id,))
