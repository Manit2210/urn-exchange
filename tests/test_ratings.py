"""Ratings: BT recovery, bootstrap coverage, duplicate combination, chip totals, outputs (offline)."""

from __future__ import annotations

import json
import math

import numpy as np

from urn_exchange.ratings import ELO_SCALE, bootstrap, bt_fit, chip_totals, compute, multi_units, write_outputs

TRUE = {"a": 1800.0, "b": 1650.0, "c": 1500.0, "d": 1450.0, "e": 1350.0, "f": 1250.0}


def _centre(d):
    m = np.mean(list(d.values()))
    return {k: v - m + 1500.0 for k, v in d.items()}


def _p(i, j, truth):
    return 1.0 / (1.0 + math.exp(-(truth[i] - truth[j]) / ELO_SCALE))


def synthetic_units(n_per_pair: int, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    truth = _centre(TRUE)
    units = []
    names = sorted(truth)
    for x, i in enumerate(names):
        for j in names[x + 1:]:
            for n in range(n_per_pair):
                s = 1.0 if rng.random() < _p(i, j, truth) else 0.0
                units.append({"unit_id": f"{i}~{j}#{n}", "block": (n, 0), "comparisons": [(i, j, s, 1.0)]})
    return units


def test_bt_recovers_known_strengths():
    truth = _centre(TRUE)
    fit = bt_fit(sorted(truth), [c for u in synthetic_units(400) for c in u["comparisons"]])
    assert abs(np.mean(list(fit.values())) - 1500.0) < 1e-6
    for k in truth:
        assert abs(fit[k] - truth[k]) < 40, (k, fit[k], truth[k])
    assert sorted(fit, key=fit.get) == sorted(truth, key=truth.get)


def test_bt_perfect_record_is_finite_and_draws_count_half():
    fit = bt_fit(["x", "y"], [("x", "y", 1.0, 1.0)] * 5)
    assert all(math.isfinite(v) for v in fit.values()) and fit["x"] > fit["y"]
    even = bt_fit(["x", "y"], [("x", "y", 0.5, 1.0)] * 10)
    assert abs(even["x"] - 1500) < 1e-6 and abs(even["y"] - 1500) < 1e-6


def test_bootstrap_intervals_contain_truth():
    truth = _centre(TRUE)
    res = bootstrap(synthetic_units(60, seed=7), sorted(truth), n_boot=300, seed=3)
    inside = sum(res[k]["lo"] <= truth[k] <= res[k]["hi"] for k in truth)
    assert inside >= len(truth) - 1, res
    assert all(res[k]["lo"] < res[k]["elo"] < res[k]["hi"] for k in truth)
    # deterministic given the seed
    again = bootstrap(synthetic_units(60, seed=7), sorted(truth), n_boot=300, seed=3)
    assert again == res


def rec(game, kind, unit, rot, seating, payoffs, deal="pilot-01", seed=0, metrics=None, status="ok"):
    return {"record": "match", "status": status, "game_id": game, "kind": kind, "unit_id": unit, "rotation": rot,
            "match_id": f"{unit}/r{rot}", "seating": list(seating), "deal_id": deal, "seed": seed,
            "result": {"payoffs": payoffs, "metrics": metrics or [{} for _ in seating], "info": {"x": 1},
                       "transcript": []},
            "steps": [{"seat": 0, "view": "v", "action": {"a": 1}, "error": None, "defaulted": False}]}


def test_multi_group_sums_and_bots_excluded():
    seats = ["a", "b", "bot:random"]
    rots = [seats[r:] + seats[:r] for r in range(3)]
    pay = [[3.0, -1.0, -2.0], [1.0, 1.0, -2.0], [0.0, 2.0, -2.0]]
    rs = [rec("p", "multi", "g0", r, rots[r], pay[r]) for r in range(3)]
    units, dropped = multi_units(rs)
    assert dropped == 0 and len(units) == 1
    chips = units[0]["chips"]
    # a sat seats 0, 2, 1 -> 3 + -2 + 2 = 3 ; b sat seats 1, 0, 2 -> -1 + 1 + -2 = -2
    assert chips == {"a": 3.0, "b": -2.0}
    assert units[0]["comparisons"] == [("a", "b", 1.0, 1.0)]
    assert multi_units(rs[:2])[1] == 1  # incomplete group dropped


def _table(deal, seed, payoffs_by_rot, models=("a", "b", "c")):
    k = len(models)
    rs = []
    for r in range(k):
        seating = list(models[r:]) + list(models[:r])
        rs.append(rec("exchange", "multi", f"exchange/{deal}/s{seed}/g0", r, seating, payoffs_by_rot[r], deal=deal,
                      seed=seed, metrics=[{"ev_chips": p / 2, "belief_logloss": 0.01 * (i + 1)}
                                          for i, p in enumerate(payoffs_by_rot[r])]))
        rs[-1]["match_id"] = f"exchange/{deal}/s{seed}/g0/r{r}"
    return rs


def test_chip_totals_and_report(tmp_path):
    rs = []
    for d in range(4):
        # seat 0 always wins 10 from seat 2: over a full rotation every model nets 0 except by deal tilt
        rs += _table(f"full-0{d + 1}", 0, [[10.0 + d, 0.0, -10.0 - d], [5.0, -5.0, 0.0], [3.0, 0.0, -3.0]])
    units, dropped = multi_units(rs)
    assert dropped == 0 and len(units) == 4
    rows = {r["model"]: r for r in chip_totals(rs, ["a", "b", "c"], n_boot=500)}
    # a: seat0 r0 (10+d), seat2 r1 (0), seat1 r2 (0) ; b: seat1 r0 (0), seat0 r1 (5), seat2 r2 (-3)
    assert rows["a"]["chips"] == sum(10.0 + d for d in range(4)) and rows["b"]["chips"] == 4 * 2.0
    assert sum(r["chips"] for r in rows.values()) == 0
    assert rows["a"]["rank"] == 1 and rows["a"]["lo"] <= rows["a"]["chips"] <= rows["a"]["hi"]
    assert rows["a"]["deals_positive"] == 4 and rows["a"]["ev_chips_per_match"] is not None
    rs.append(rec("exchange", "multi", "exchange/full-09/s0/g0", 0, ["a", "b", "c"], [0.0, 0.0, 0.0], status="error"))
    meta = {"exchange": {"kind": "multi", "headline": {"chips": "higher_is_better",
                                                       "belief_logloss": "lower_is_better"}}}
    res = compute(rs, meta, n_boot=50)
    g = res["games"]["exchange"]
    assert g["matches_not_ok"] == 1 and g["chips"][0]["model"] == "a"
    bt = next(b for b in g["boards"] if b["board"] == "bt")
    assert {r["model"]: r["rank"] for r in bt["rows"]}["a"] == 1 and bt["units"] == 4
    files = write_outputs(res, tmp_path / "rep")
    json.loads((tmp_path / "rep" / "ratings.json").read_text())  # strict JSON (no NaN)
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "**chips**" in md and "Skill profile" in md and "Highlights" in md
    assert set(files) == {"ratings.json", "ratings.csv", "chips.csv", "profile.csv", "report.md"}
