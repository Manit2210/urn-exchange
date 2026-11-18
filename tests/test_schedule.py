"""Schedule: counts, mirrored pairs, balanced multi groups, determinism (offline)."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import pytest

from urn_exchange.schedule import (BOT_PREFIX, Config, GameEntry, GameInfo, balanced_groups, build_schedule,
                                   select_deals, summarize)

DEALS = tuple([f"pilot-{i:02d}" for i in range(1, 7)] + [f"full-{i:02d}" for i in range(1, 25)])
CAT = {
    "s": GameInfo("s", "1", "solo", 1, 1, DEALS),
    "d": GameInfo("d", "1", "duel", 2, 2, DEALS),
    "m": GameInfo("m", "1", "multi", 4, 5, DEALS),
}


def cfg(n_models: int, games: list[dict]) -> Config:
    return Config.model_validate({
        "budget_usd": 0, "models": [{"key": f"m{i}", "provider": "fake", "model": "fake-model"} for i in range(n_models)],
        "games": games})


def test_solo_and_duel_counts_and_mirrors():
    c = cfg(4, [{"game": "s", "n_deals": 3, "seeds": [0, 1]}, {"game": "d", "n_deals": 3, "seeds": [0, 1]}])
    sched = build_schedule(c, CAT)
    solo = [m for m in sched if m.kind == "solo"]
    duel = [m for m in sched if m.kind == "duel"]
    assert len(solo) == 4 * 3 * 2
    assert len(duel) == math.comb(4, 2) * 3 * 2 * 2
    pairs: dict[str, list] = {}
    for m in duel:
        pairs.setdefault(m.unit_id, []).append(m)
    assert len(pairs) == math.comb(4, 2) * 3 * 2
    for ms in pairs.values():
        a, b = sorted(ms, key=lambda m: m.rotation)
        assert (a.deal_id, a.seed) == (b.deal_id, b.seed)
        assert a.seating == tuple(reversed(b.seating)) and a.rotation == 0 and b.rotation == 1
    # Every solo match on the same deal shares a unit id (paired by deal across models).
    assert len({m.unit_id for m in solo}) == 3 * 2
    assert all(m.deal_id.startswith("pilot-") for m in sched)


def test_multi_rotation_and_balance():
    k, n, d = 5, 10, 6
    c = cfg(n, [{"game": "m", "n_deals": d, "n_players": k}])
    sched = build_schedule(c, CAT)
    assert len(sched) == d * (n // k) * k
    groups: dict[str, list] = {}
    for m in sched:
        groups.setdefault(m.unit_id, []).append(m)
    for ms in groups.values():
        members = set(ms[0].seating)
        assert len(ms) == k and all(set(m.seating) == members for m in ms)
        for seat in range(k):  # every model sits in every seat once (duplicate rotation)
            assert {m.seating[seat] for m in ms} == members
    seats = Counter(k_ for m in sched for k_ in m.models())
    assert set(seats.values()) == {d * k}  # each model once per deal, k rotations


def test_multi_pair_balance_when_not_divisible():
    k, n, blocks = 4, 10, 30
    counts: dict = {}
    models = [f"m{i}" for i in range(n)]
    for b in range(blocks):
        groups = balanced_groups(models, k, f"blk{b}", counts)
        assert len(groups) == 3 and all(len(set(g)) == k for g in groups)
        assert set().union(*map(set, groups)) == set(models)  # everyone plays every block
    app = counts["appearances"]
    assert max(app.values()) - min(app.values()) <= 1
    pairs = [counts["pairs"][frozenset(p)] for p in combinations(models, 2)]
    assert min(pairs) >= 1
    mean = sum(pairs) / len(pairs)
    assert max(pairs) - min(pairs) <= max(4, 0.5 * mean), (min(pairs), max(pairs), mean)


def test_fewer_models_than_seats_pads_with_bots():
    c = cfg(2, [{"game": "m", "n_deals": 1, "n_players": 4, "fill_bot": "random"}])
    sched = build_schedule(c, CAT)
    assert len(sched) == 4
    for m in sched:
        assert sum(s.startswith(BOT_PREFIX) for s in m.seating) == 2
        assert sorted(m.models()) == ["m0", "m1"]


def test_deterministic_and_unique():
    c = cfg(6, [{"game": "s", "n_deals": 2}, {"game": "d", "n_deals": 2}, {"game": "m", "n_deals": 4, "n_players": 4}])
    a, b = build_schedule(c, CAT), build_schedule(c, CAT)
    assert a == b
    assert len({m.match_id for m in a}) == len(a)


def test_deal_selection_and_errors():
    info = CAT["s"]
    assert select_deals(info, GameEntry(game="s", split="full", n_deals=2)) == ["full-01", "full-02"]
    assert select_deals(info, GameEntry(game="s", deals=["full-03"])) == ["full-03"]
    assert len(select_deals(info, GameEntry(game="s", split="all"))) == len(DEALS)
    with pytest.raises(KeyError):
        select_deals(info, GameEntry(game="s", deals=["nope"]))
    with pytest.raises(ValueError):
        build_schedule(cfg(6, [{"game": "m", "n_players": 6}]), CAT)
    with pytest.raises(KeyError):
        build_schedule(cfg(2, [{"game": "zzz"}]), CAT)


def test_slices_and_summary():
    c = cfg(4, [{"game": "s", "n_deals": 2}, {"game": "d", "n_deals": 2}])
    sched = build_schedule(c, CAT, models=["m0", "m1"], games=["d"])
    assert {m.game_id for m in sched} == {"d"} and len(sched) == 2 * 2
    s = summarize(sched, {"d": 5.0})
    assert s.matches == 4 and s.by_game["d"]["units"] == 2 and s.seats_by_model == {"m0": 4, "m1": 4}
    assert s.est_calls == 4 * 2 * 5
    with pytest.raises(KeyError):
        build_schedule(c, CAT, models=["nobody"])


def test_shared_params_override_model_params():
    c = Config.model_validate({"budget_usd": 0, "params": {"max_tokens": 16000, "reasoning_effort": "high"},
                                    "models": [{"key": "a", "provider": "fake", "model": "fake-model",
                                                "params": {"reasoning_effort": "low", "cache_control": {"type": "x"}}}],
                                    "games": [{"game": "s"}]})
    p = c.effective_models()[0].params
    assert p == {"max_tokens": 16000, "reasoning_effort": "high", "cache_control": {"type": "x"}}
    with pytest.raises(ValueError):
        Config.model_validate({"budget_usd": 0, "games": [], "models": [
            {"key": "a", "provider": "fake", "model": "x"}, {"key": "a", "provider": "fake", "model": "y"}]})


def test_rotations_limit_and_v1_seating():
    c = cfg(6, [{"game": "m", "n_deals": 1, "n_players": 5, "rotations": 2}])
    assert len(build_schedule(c, CAT)) == 2 * 2  # 2 groups (6 models, tables of 5) x 2 rotations
    six = {**CAT, "m": GameInfo("m", "1", "multi", 3, 6, DEALS)}
    c = cfg(6, [{"game": "m", "split": "full", "n_deals": 6, "seeds": [0, 1], "n_players": 6}])
    sched = build_schedule(c, six)
    assert len(sched) == 72 and len({m.unit_id for m in sched}) == 12
    first = [m for m in sched if m.deal_id == "full-02" and m.seed == 0 and m.rotation == 1][0]
    assert first.seating == ("m1", "m2", "m3", "m4", "m5", "m0")  # roster order, rotated by one seat
