"""Ratings: Bradley-Terry Elo with bootstrap intervals, chip totals, skill profile and highlights.

Reads only a run's match records (``matches.jsonl`` or a trace file) and its manifest, never the spend
ledger. Definitions (docs/harness.md, section "Ratings"):

* **Chips.** Each model's chips summed over every match, with a 95% percentile bootstrap range over
  duplicate deals (one deal = one (deal, seed) block played in every seat rotation; 4,000 resamples of
  the per-deal totals, fixed seed). This is the headline ranking.
* **Elo (BT).** One duplicate group = the same deal with the models rotated through every seat. Each
  model's chips are summed over the group's rotations, then every pair of models in the group is one
  comparison (win/draw/loss on those sums; margins do not enter the fit). Bot seats are not rated.
  Incomplete groups are dropped.
* **Fit.** Penalised maximum likelihood, P(i beats j) = 1 / (1 + exp(-(t_i - t_j))), draws count half,
  with a weak Gaussian prior on t (SD 600 Elo) so a perfect record stays finite; Elo = 1500 +
  400/ln(10) * (t - mean t), so 1500 is the roster mean.
* **Intervals.** Percentile bootstrap (default 1000 resamples, fixed seed) over groups, refitting each
  time. ``cluster="deal"`` resamples whole (deal, seed) blocks instead (the same thing at one table).
* **Skill profile.** Per model, the mean of each headline metric (``HEADLINE`` on the game: chips,
  belief_logloss, market_read, trade_regret) with a bootstrap range over deals, and the model's rank.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from urn_exchange.schedule import BOT_PREFIX

ELO_SCALE = 400.0 / math.log(10.0)
PRIOR_SD_ELO = 600.0
DEFAULT_SEED = 20261116
DRAW_EPS = 1e-9


# --------------------------------------------------------------------------- Bradley-Terry


def bt_fit(players: list[str], comparisons: Iterable[tuple[str, str, float, float]], *,
           prior_sd_elo: float = PRIOR_SD_ELO, iters: int = 100, tol: float = 1e-10) -> dict[str, float]:
    """Penalised BT by Newton's method. ``comparisons`` = (i, j, score of i in [0,1], weight).
    Returns Elo per player (1500 = mean of ``players``)."""
    idx = {p: n for n, p in enumerate(players)}
    comps = [(idx[i], idx[j], float(s), float(w)) for i, j, s, w in comparisons if i in idx and j in idx]
    n = len(players)
    if n == 0:
        return {}
    lam = (ELO_SCALE / prior_sd_elo) ** 2
    theta = np.zeros(n)
    if comps:
        I = np.array([c[0] for c in comps])
        J = np.array([c[1] for c in comps])
        S = np.array([c[2] for c in comps])
        W = np.array([c[3] for c in comps])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(theta[I] - theta[J])))
            r = W * (S - p)
            g = np.bincount(I, r, n) - np.bincount(J, r, n) - lam * theta
            h = W * p * (1 - p)
            H = np.zeros((n, n))
            np.add.at(H, (I, I), -h)
            np.add.at(H, (J, J), -h)
            np.add.at(H, (I, J), h)
            np.add.at(H, (J, I), h)
            H -= lam * np.eye(n)
            step = np.linalg.solve(H, g)
            theta = theta - step
            if float(np.max(np.abs(step))) < tol:
                break
    theta = theta - theta.mean()
    return {p: float(1500.0 + ELO_SCALE * theta[idx[p]]) for p in players}


def _score(a: float, b: float) -> float:
    return 0.5 if abs(a - b) <= DRAW_EPS else (1.0 if a > b else 0.0)


def bootstrap(units: list[dict], players: list[str], *, n_boot: int = 1000, seed: int = DEFAULT_SEED,
              alpha: float = 0.05, cluster: str = "unit") -> dict[str, dict]:
    """Point fit plus percentile bootstrap. Each unit: {unit_id, block, comparisons: [(i, j, s, w)]}.
    ``cluster`` = "unit" (pairs/groups) or "deal" (whole (deal, seed) blocks)."""
    point = bt_fit(players, [c for u in units for c in u["comparisons"]])
    games = defaultdict(int)
    for u in units:
        for i, j, _s, _w in u["comparisons"]:
            games[i] += 1
            games[j] += 1
    out = {p: {"elo": point[p], "lo": float("nan"), "hi": float("nan"), "n": games[p]} for p in players}
    if not units or n_boot <= 0:
        return out
    groups: dict[Any, list[dict]] = defaultdict(list)
    for u in units:
        groups[u["block"] if cluster == "deal" else u["unit_id"]].append(u)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = {p: [] for p in players}
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        comps = [c for k in pick for u in groups[keys[k]] for c in u["comparisons"]]
        fit = bt_fit(players, comps)
        for p in players:
            draws[p].append(fit[p])
    for p in players:
        lo, hi = np.percentile(draws[p], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[p]["lo"], out[p]["hi"] = float(lo), float(hi)
    return out


# --------------------------------------------------------------------------- records -> units


def ok_records(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if r.get("record") == "match" and r.get("status") == "ok"]


def _payoff(r: dict, seat: int) -> float:
    return float(r["result"]["payoffs"][seat])


def multi_units(records: list[dict], all_records: list[dict] | None = None) -> tuple[list[dict], int]:
    """One unit per complete duplicate group; pairwise comparisons on chips summed over the rotations.
    A group is complete when every rotation 0..k-1 finished ok."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["unit_id"]].append(r)
    units, dropped = [], 0
    for uid, rs in sorted(by.items()):
        k = len(rs[0]["seating"])
        rots = {r["rotation"]: r for r in rs}
        if sorted(rots) != list(range(k)):
            dropped += 1
            continue
        chips: dict[str, float] = defaultdict(float)
        for r in rots.values():
            for seat, key in enumerate(r["seating"]):
                if not key.startswith(BOT_PREFIX):
                    chips[key] += _payoff(r, seat)
        models = sorted(chips)
        comps = [(a, b, _score(chips[a], chips[b]), 1.0) for n, a in enumerate(models) for b in models[n + 1:]]
        units.append({"unit_id": uid, "block": (rs[0]["deal_id"], rs[0]["seed"]), "comparisons": comps,
                      "chips": dict(chips)})
    return units, dropped


def chip_totals(records: list[dict], models: list[str], *, n_boot: int = 4000, seed: int = 1) -> list[dict]:
    """Total chips per model with a 95% bootstrap range over duplicate deals (per-deal totals resampled with
    replacement, ``random.Random(seed)``), plus luck-free ``ev_chips`` per match."""
    import random

    by_deal: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ev: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for seat, m in enumerate(r["seating"]):
            if m.startswith(BOT_PREFIX):
                continue
            by_deal[r["unit_id"]][m] += _payoff(r, seat)
            v = seat_metric(r, seat, "ev_chips")
            if v is not None:
                ev[m].append(v)
    deals = sorted(by_deal)
    rows = []
    for m in models:
        per = [by_deal[u][m] for u in deals if m in by_deal[u]]
        lo = hi = float("nan")
        if per and n_boot > 0:
            rng = random.Random(seed)
            sums = sorted(sum(rng.choice(per) for _ in per) for _ in range(n_boot))
            lo, hi = sums[int(0.025 * n_boot)], sums[int(0.975 * n_boot)]
        rows.append({"model": m, "chips": float(sum(per)), "lo": float(lo), "hi": float(hi), "deals": len(per),
                     "deals_positive": sum(1 for x in per if x > 0),
                     "ev_chips_per_match": float(np.mean(ev[m])) if ev[m] else None})
    _rank(rows, "chips", "higher_is_better")
    return rows


# --------------------------------------------------------------------------- metrics


def headline_of(meta: dict | None) -> dict[str, str]:
    h = (meta or {}).get("headline")
    return dict(h) if h else {"chips": "higher_is_better"}


def seat_metric(r: dict, seat: int, key: str) -> float | None:
    if key == "chips":
        return _payoff(r, seat)
    metrics = r["result"].get("metrics") or []
    v = metrics[seat].get(key) if seat < len(metrics) and isinstance(metrics[seat], dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def model_seat_values(records: list[dict], key: str) -> dict[str, list[tuple[Any, float]]]:
    """model -> [(block, value)] over every model seat in ``records``."""
    out: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    for r in records:
        for seat, m in enumerate(r["seating"]):
            if m.startswith(BOT_PREFIX):
                continue
            v = seat_metric(r, seat, key)
            if v is not None:
                out[m].append(((r["deal_id"], r["seed"]), v))
    return out


def mean_ci(vals: list[tuple[Any, float]], *, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Mean with a percentile bootstrap over blocks (deals)."""
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    xs = np.array([v for _, v in vals], dtype=float)
    out = {"mean": float(xs.mean()), "lo": float("nan"), "hi": float("nan"), "n": len(xs)}
    blocks: dict[Any, list[float]] = defaultdict(list)
    for b, v in vals:
        blocks[b].append(v)
    keys = sorted(blocks)
    if n_boot > 0 and len(keys) > 1:
        rng = np.random.default_rng(seed)
        ms = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            ms.append(float(np.mean([v for k in pick for v in blocks[keys[k]]])))
        lo, hi = np.percentile(ms, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["lo"], out["hi"] = float(lo), float(hi)
    return out


def _rank(rows: list[dict], value_key: str, direction: str) -> None:
    present = [r for r in rows if r.get(value_key) is not None]
    present.sort(key=lambda r: r[value_key], reverse=direction != "lower_is_better")
    for n, r in enumerate(present, 1):
        r["rank"] = n


# --------------------------------------------------------------------------- report


def load_runs(run_dirs: list[str | Path]) -> tuple[list[dict], dict]:
    """Records (later sources win for a repeated match id) and games meta. A source is a run directory
    (``matches.jsonl`` + ``manifest.json``) or a trace file (``.jsonl`` / ``.jsonl.gz``); without a manifest
    the meta comes from the game classes this package ships."""
    from urn_exchange.runner import default_registry, game_meta, latest_by_match, load_records

    recs: dict[str, dict] = {}
    meta: dict[str, dict] = {g: game_meta(c) for g, c in default_registry()[0].items()}
    for d in run_dirs:
        m = Path(d) / "manifest.json"
        if Path(d).is_dir() and m.exists():
            meta.update(json.loads(m.read_text()).get("games") or {})
        recs.update(latest_by_match(load_records(d)))
    return list(recs.values()), meta


def compute(records: list[dict], games_meta: dict, *, n_boot: int = 1000, seed: int = DEFAULT_SEED,
            cluster: str = "unit", top_n: int = 3) -> dict:
    ok = ok_records(records)
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_game[r["game_id"]].append(r)
    errors = defaultdict(int)
    for r in records:
        if r.get("record") == "match" and r.get("status") != "ok":
            errors[r["game_id"]] += 1
    out: dict[str, Any] = {"games": {}, "profile": [], "highlights": {}, "n_boot": n_boot, "seed": seed,
                           "cluster": cluster}
    for gid in sorted(set(by_game) | set(errors)):
        rs = by_game.get(gid, [])
        meta = games_meta.get(gid) or {}
        kind = meta.get("kind") or (rs[0]["kind"] if rs else None)
        g: dict[str, Any] = {"kind": kind, "matches_ok": len(rs), "matches_not_ok": errors.get(gid, 0),
                             "headline": headline_of(meta), "boards": []}
        models = sorted({m for r in rs for m in r["seating"] if not m.startswith(BOT_PREFIX)})
        if kind == "multi":
            units, dropped = multi_units(rs)
            fit = bootstrap(units, models, n_boot=n_boot, seed=seed, cluster=cluster)
            rows = [{"model": m, **fit[m]} for m in models]
            _rank(rows, "elo", "higher_is_better")
            g["boards"].append({"board": "bt", "metric": "elo", "units": len(units), "dropped_units": dropped,
                                "rows": rows})
            g["chips"] = chip_totals(rs, models)
        # The headline metrics of every game for the profile.
        for key, direction in headline_of(meta).items():
            vals = model_seat_values(rs, key)
            rows = [{"model": m, **mean_ci(vals.get(m, []), n_boot=n_boot, seed=seed)} for m in models]
            _rank(rows, "mean", direction)
            for row in rows:
                out["profile"].append({"model": row["model"], "game": gid, "metric": key, "direction": direction,
                                       "value": row["mean"], "lo": row["lo"], "hi": row["hi"], "n": row["n"],
                                       "rank": row.get("rank")})
        out["games"][gid] = g
        out["highlights"][gid] = highlights(rs, meta, top_n=top_n)
    return out


def _short(obj: Any, n: int = 220) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def highlights(records: list[dict], meta: dict, *, top_n: int = 3, excerpt_steps: int = 4) -> list[dict]:
    """Top matches by chip swing (max - min payoff; |payoff| for solo) with a short transcript excerpt, plus
    the largest belief errors when a headline metric names a belief."""
    out = []
    swing = sorted(records, key=lambda r: (-(max(r["result"]["payoffs"]) - min(r["result"]["payoffs"])
                                            if len(r["seating"]) > 1 else -abs(r["result"]["payoffs"][0])),
                                           r["match_id"]))
    for r in swing[:top_n]:
        steps = [s for s in r.get("steps", []) if not s.get("defaulted")][-excerpt_steps:]
        out.append({"type": "chip_swing", "match_id": r["match_id"], "seating": r["seating"],
                    "payoffs": r["result"]["payoffs"],
                    "excerpt": [{"seat": s["seat"], "model": r["seating"][s["seat"]], "action": _short(s["action"]),
                                 **({"error": s["error"]} if s.get("error") else {})} for s in steps],
                    "revealed": _short(r["result"].get("info") or {}, 300)})
    for key, direction in headline_of(meta).items():
        if "belief" not in key or direction != "lower_is_better":
            continue
        seats = [(v, r, s) for r in records for s, m in enumerate(r["seating"])
                 if not m.startswith(BOT_PREFIX) and (v := seat_metric(r, s, key)) is not None]
        for v, r, s in sorted(seats, key=lambda x: (-x[0], x[1]["match_id"]))[:top_n]:
            out.append({"type": f"largest_{key}", "match_id": r["match_id"], "model": r["seating"][s], "seat": s,
                        "value": v})
    return out


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def to_markdown(res: dict) -> str:
    lines = ["# Urn Exchange ratings", "",
             "Paper chips, research and fun only; not investment advice. Definitions: docs/harness.md (Ratings).", "",
             f"Bootstrap: {res['n_boot']} resamples over {res['cluster']}s, 95% percentile intervals.", ""]
    for gid, g in res["games"].items():
        lines += [f"## {gid} ({g['kind']})", "",
                  f"Matches ok: {g['matches_ok']}; not ok: {g['matches_not_ok']}.", ""]
        if g.get("chips"):
            lines += ["**chips** (total over all matches; 95% bootstrap range over duplicate deals; ev = luck-free "
                      "chips per match)", "",
                      "| rank | model | chips | 95% range | deals > 0 | ev per match |", "|---|---|---|---|---|---|"]
            for row in sorted(g["chips"], key=lambda r: r.get("rank", 99)):
                lines.append(f"| {row.get('rank', '-')} | {row['model']} | {row['chips']:+,.0f} | "
                             f"{_fmt(row['lo'], 0)} to {_fmt(row['hi'], 0)} | {row['deals_positive']}/{row['deals']} | "
                             f"{_fmt(row['ev_chips_per_match'], 1)} |")
            lines.append("")
        for b in g["boards"]:
            if b["metric"] == "elo":
                lines += [f"**{b['board']}** ({b.get('units', 0)} units"
                          + (f", {b['dropped_units']} dropped" if b.get("dropped_units") else "") + ")", "",
                          "| rank | model | Elo | 95% CI | comparisons |", "|---|---|---|---|---|"]
                for row in sorted(b["rows"], key=lambda r: r.get("rank", 99)):
                    lines.append(f"| {row.get('rank', '-')} | {row['model']} | {_fmt(row['elo'], 0)} | "
                                 f"{_fmt(row['lo'], 0)} to {_fmt(row['hi'], 0)} | {row['n']} |")
            else:
                lines += [f"**{b['board']}** ({b['direction'].replace('_', ' ')})", "",
                          "| rank | model | mean | 95% CI | seats |", "|---|---|---|---|---|"]
                for row in sorted(b["rows"], key=lambda r: r.get("rank", 99)):
                    lines.append(f"| {row.get('rank', '-')} | {row['model']} | {_fmt(row['mean'], 3)} | "
                                 f"{_fmt(row['lo'], 3)} to {_fmt(row['hi'], 3)} | {row['n']} |")
            lines.append("")
    cols = sorted({(p["game"], p["metric"]) for p in res["profile"]})
    models = sorted({p["model"] for p in res["profile"]})
    if cols:
        cell = {(p["model"], p["game"], p["metric"]): p for p in res["profile"]}
        lines += ["## Skill profile", "", "Mean of each game's headline metric; rank in brackets.", "",
                  "| model | " + " | ".join(f"{g}: {m}" for g, m in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for m in models:
            vals = []
            for g, k in cols:
                p = cell.get((m, g, k))
                vals.append("-" if not p or p["value"] is None else f"{p['value']:.3g} ({p.get('rank', '-')})")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
        lines.append("")
    lines += ["## Highlights", ""]
    for gid, hs in res["highlights"].items():
        for h in hs:
            if h["type"] == "chip_swing":
                lines.append(f"- **{gid}** `{h['match_id']}` payoffs {h['payoffs']}; revealed: {h['revealed']}")
                for s in h["excerpt"]:
                    lines.append(f"  - seat {s['seat']} ({s['model']}): `{s['action']}`"
                                 + (f" ERROR {s['error']}" if s.get("error") else ""))
            else:
                lines.append(f"- **{gid}** {h['type']}: {h['model']} seat {h['seat']} = {h['value']:.4g} "
                             f"(`{h['match_id']}`)")
    return "\n".join(lines) + "\n"


def write_outputs(res: dict, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ratings.json").write_text(json.dumps(clean(res), indent=2, sort_keys=True, allow_nan=False))
    with (out / "ratings.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "board", "model", "rank", "value", "lo", "hi", "n"])
        for gid, g in res["games"].items():
            for b in g["boards"]:
                for row in b["rows"]:
                    w.writerow([gid, b["board"], row["model"], row.get("rank"), row.get("elo", row.get("mean")),
                                row.get("lo"), row.get("hi"), row.get("n")])
    with (out / "profile.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "game", "metric", "direction", "value", "lo", "hi", "n", "rank"])
        for p in res["profile"]:
            w.writerow([p[k] for k in ("model", "game", "metric", "direction", "value", "lo", "hi", "n", "rank")])
    with (out / "chips.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "model", "rank", "chips", "lo", "hi", "deals", "deals_positive", "ev_chips_per_match"])
        for gid, g in res["games"].items():
            for row in g.get("chips") or []:
                w.writerow([gid] + [row.get(k) for k in ("model", "rank", "chips", "lo", "hi", "deals",
                                                         "deals_positive", "ev_chips_per_match")])
    (out / "report.md").write_text(to_markdown(res))
    return {k: str(out / k) for k in ("ratings.json", "ratings.csv", "chips.csv", "profile.csv", "report.md")}


def clean(o: Any) -> Any:
    """Strict-JSON copy: NaN/inf -> None, tuples -> lists, numpy scalars -> Python numbers."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def report(run_dirs: list[str | Path], out_dir: str | Path, *, n_boot: int = 1000, seed: int = DEFAULT_SEED,
           cluster: str = "unit") -> dict:
    records, meta = load_runs(run_dirs)
    res = compute(records, meta, n_boot=n_boot, seed=seed, cluster=cluster)
    res["sources"] = [Path(d).name for d in run_dirs]
    files = write_outputs(res, out_dir)
    return {"files": files, "games": {g: {"matches_ok": v["matches_ok"], "boards": [b["board"] for b in v["boards"]]}
                                      for g, v in res["games"].items()}}
