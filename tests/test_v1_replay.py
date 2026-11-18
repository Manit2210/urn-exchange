"""The game in this repository is the game v1 was played on: replaying v1 moves reproduces every view, error,
system prompt, payoff and metric. One committed match always; every match when the trace file is present."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from urn_exchange import exchange as X
from urn_exchange.core import play_match
from urn_exchange.runner import load_records, system_prompt

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces" / "urn-exchange-v1-traces.jsonl.gz"


class Replay:
    def __init__(self, actions: list[dict]):
        self.name = "replay"
        self.queue = list(actions)

    def act(self, view, schema):
        return self.queue.pop(0)

    def feedback(self, error):
        pass


def close(a, b) -> bool:
    """Equal, with floats compared to 1e-9 (the last digit of a logarithm can differ between platforms)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def replay(deal: str, seed: int, actions: dict[int, list[dict]]):
    g = X.Exchange()
    log = play_match(g, [Replay(actions[s]) for s in range(len(actions))], deal, seed)
    return g, log


def test_committed_v1_match_replays_exactly():
    fx = json.loads((ROOT / "tests" / "fixtures" / "v1_match.json").read_text())
    g, log = replay(fx["deal_id"], fx["seed"], {int(s): a for s, a in fx["actions"].items()})
    assert [s["error"] for s in log.steps] == fx["errors"]
    assert hashlib.sha256(json.dumps([s["view"] for s in log.steps]).encode()).hexdigest() == fx["views_sha256"]
    assert [hashlib.sha256(system_prompt(g, s).encode()).hexdigest() for s in range(6)] == fx["system_sha256"]
    assert log.result.payoffs == fx["payoffs"]
    assert close(json.loads(json.dumps(log.result.metrics)), fx["metrics"])


@pytest.mark.skipif(not TRACES.exists(), reason="trace file not downloaded (GitHub release attachment)")
def test_every_v1_trace_replays_exactly():
    recs = load_records(TRACES)
    assert {r["dataset"] for r in recs} == {"v1", "v1-xhigh"} and len(recs) == 78
    for r in recs:
        n = len(r["seating"])
        acts = {s: [st["action"] for st in r["steps"] if st["seat"] == s and not st["defaulted"]] for s in range(n)}
        g, log = replay(r["deal_id"], r["seed"], acts)
        assert [(s["seat"], s["view"], s["error"]) for s in log.steps] == \
            [(s["seat"], s["view"], s["error"]) for s in r["steps"]], r["match_id"]
        assert log.result.payoffs == r["result"]["payoffs"]
        assert close(json.loads(json.dumps(log.result.metrics)), r["result"]["metrics"])
        assert close(json.loads(json.dumps(log.result.info)), r["result"]["info"])
        assert all(system_prompt(g, s) == r["players"][s]["system"] for s in range(n))
