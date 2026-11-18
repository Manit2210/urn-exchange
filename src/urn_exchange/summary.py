"""Results summary: one CSV row per (match, seat) and a per-model summary JSON.

``matches.csv`` columns: dataset, match id, deal, seed, family, rotation, seat, model, chips (the seat's
payoff), ev_chips (fills valued at the pooled posterior of every player's balls, net of research fees: chips
with the urn luck removed), the graded metrics of ``Exchange._seat_metrics`` and per-seat call totals
(calls, tokens, cost, median latency).

``summary.json`` per model: chips with a 95% bootstrap range over duplicate deals, luck-free ev per match,
deals won and positive, chips and ev by deal family, behaviour (pass rates, sizes, research rate, notes),
calls, output tokens per move, latency and cost, and the BT Elo when a ratings result is given.
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from urn_exchange.exchange import FAMILY_CYCLE
from urn_exchange.ratings import chip_totals
from urn_exchange.schedule import BOT_PREFIX

METRICS = ("ev_chips", "final_chips", "belief_logloss", "belief_brier", "market_read", "market_gain",
           "r2_belief_vs_own", "trade_regret", "trade_regret_r1", "trade_regret_r2", "trade_regret_r2_own",
           "wrong_direction", "pass_when_should_trade", "trade_when_should_pass", "draws", "fees_paid",
           "fees_received", "trades", "volume", "decisions", "insider")
CALL_COLS = ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "cost_usd", "latency_s_median")


def _player_calls(rec: dict, seat: int) -> list[dict]:
    for p in rec.get("players") or []:
        if p.get("seat") == seat:
            return [c for c in p.get("calls") or [] if "error" not in c]
    return []


def match_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in sorted(records, key=lambda x: (x.get("dataset", ""), x["match_id"])):
        if r.get("status") != "ok":
            continue
        info = r["result"].get("info") or {}
        for seat, model in enumerate(r["seating"]):
            met = (r["result"].get("metrics") or [{}])[seat] or {}
            calls = _player_calls(r, seat)
            lat = [float(c.get("latency_s") or 0.0) for c in calls]
            row: dict[str, Any] = {
                "dataset": r.get("dataset", ""), "match_id": r["match_id"], "deal": r["deal_id"], "seed": r["seed"],
                "family": info.get("family"), "rotation": r.get("rotation"), "seat": seat + 1, "model": model,
                "chips": r["result"]["payoffs"][seat],
            }
            for k in METRICS:
                row[k] = met.get(k)
            row.update({
                "calls": len(calls),
                "input_tokens": sum(int((c.get("usage") or {}).get("input_tokens", 0)) for c in calls),
                "cached_input_tokens": sum(int((c.get("usage") or {}).get("cached_input_tokens", 0)) for c in calls),
                "output_tokens": sum(int((c.get("usage") or {}).get("output_tokens", 0)) for c in calls),
                "cost_usd": round(sum(float(c.get("cost_usd") or 0.0) for c in calls), 6),
                "latency_s_median": round(st.median(lat), 2) if lat else None,
            })
            rows.append(row)
    return rows


def write_match_csv(records: list[dict], path: str | Path) -> int:
    rows = match_rows(records)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "match_id", "deal", "seed", "family", "rotation", "seat", "model", "chips", *METRICS, *CALL_COLS]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
    return len(rows)


def summarize(records: list[dict], ratings: dict | None = None, names: dict[str, str] | None = None) -> dict:
    """Per-model summary of the ok matches in ``records`` (one dataset)."""
    ms = [r for r in records if r.get("status") == "ok" and r.get("record", "match") == "match"]
    models = sorted({k for m in ms for k in m["seating"] if not k.startswith(BOT_PREFIX)})
    deals = sorted({m["unit_id"] for m in ms})
    by_deal: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ev_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    forced: dict[str, int] = defaultdict(int)
    r1 = defaultdict(lambda: {"orders": 0, "passes": 0, "size": [], "max": 0})
    r2 = defaultdict(lambda: {"orders": 0, "passes": 0})
    research = defaultdict(lambda: {"offers": 0, "bought": 0})
    notes: dict[str, int] = defaultdict(int)
    calls: dict[str, list[tuple[float, int, float]]] = defaultdict(list)
    wall = []
    family_of = {}
    for m in ms:
        fam = m["result"]["info"]["family"]
        family_of[m["unit_id"]] = (m["deal_id"], m["seed"], fam)
        wall.append(float(m.get("wall_s") or 0.0))
        for s, k in enumerate(m["seating"]):
            if k.startswith(BOT_PREFIX):
                continue
            p = m["result"]["payoffs"][s]
            met = m["result"]["metrics"][s] or {}
            by_deal[m["unit_id"]][k] += p
            by_family[fam][k] += p
            ev_family[fam][k] += met.get("ev_chips") or 0.0
            forced[k] += (m.get("defaulted") or [0] * len(m["seating"]))[s]
            for key in ("ev_chips", "fees_paid", "fees_received", "trades", "volume", "wrong_direction",
                        "pass_when_should_trade", "trade_when_should_pass", "market_read", "belief_logloss",
                        "trade_regret"):
                v = met.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric[k][key].append(float(v))
        for step in m.get("steps") or []:
            if step.get("error") is not None or step.get("defaulted"):
                continue
            k = m["seating"][step["seat"]]
            a = step.get("action") or {}
            view = step.get("view") or ""
            if "research" in a:
                research[k]["offers"] += 1
                research[k]["bought"] += bool(a["research"])
            elif a.get("side") in ("buy", "sell", "pass"):
                if "round 1 of 2" in view:
                    r1[k]["orders"] += 1
                    r1[k]["passes"] += a["side"] == "pass"
                    if a["side"] != "pass":
                        r1[k]["size"].append(a.get("size") or 0)
                        r1[k]["max"] += (a.get("size") or 0) >= 10
                elif "round 2 of 2" in view:
                    r2[k]["orders"] += 1
                    r2[k]["passes"] += a["side"] == "pass"
            if a.get("note"):
                notes[k] += 1
        for pl in m.get("players") or []:
            k = pl.get("model_key")
            if not k:
                continue
            for c in pl.get("calls") or []:
                if "error" in c:
                    continue
                u = c.get("usage") or {}
                calls[k].append((float(c.get("latency_s") or 0.0), int(u.get("output_tokens") or 0),
                                 float(c.get("cost_usd") or 0.0)))
    chips = {row["model"]: row for row in chip_totals(ms, models)}
    elo = {}
    if ratings:
        for g in (ratings.get("games") or {}).values():
            for b in g.get("boards") or []:
                if b.get("board") == "bt":
                    elo = {r["model"]: r for r in b["rows"]}

    def mean(k: str, key: str) -> float | None:
        return round(st.mean(metric[k][key]), 4) if metric[k][key] else None

    n_matches = len(ms)
    out_models = []
    for k in sorted(models, key=lambda x: -chips[x]["chips"]):
        lat = [c[0] for c in calls[k]]
        tok = [c[1] for c in calls[k]]
        cost = sum(c[2] for c in calls[k])
        seats = sum(1 for m in ms for s in m["seating"] if s == k)
        row: dict[str, Any] = {
            "key": k, "name": (names or {}).get(k, k),
            "chips": round(chips[k]["chips"]), "chips_ci95": [round(chips[k]["lo"]), round(chips[k]["hi"])],
            "ev_chips_per_match": round(st.mean(metric[k]["ev_chips"]), 1) if metric[k]["ev_chips"] else None,
            "deals_won": sum(1 for u in deals if by_deal[u][k] == max(by_deal[u].values())),
            "deals_positive": sum(1 for u in deals if by_deal[u][k] > 0),
            "forced_passes": forced[k],
            "per_deal": [round(by_deal[u][k]) for u in deals],
            "by_family": {f: round(by_family[f][k]) for f in FAMILY_CYCLE if f in by_family},
            "ev_by_family": {f: round(ev_family[f][k]) for f in FAMILY_CYCLE if f in ev_family},
            "behaviour": {
                "r1_pass_rate": round(r1[k]["passes"] / r1[k]["orders"], 3) if r1[k]["orders"] else None,
                "r2_pass_rate": round(r2[k]["passes"] / r2[k]["orders"], 3) if r2[k]["orders"] else None,
                "r1_avg_size": round(st.mean(r1[k]["size"]), 2) if r1[k]["size"] else None,
                "r1_max_size_rate": round(r1[k]["max"] / max(1, len(r1[k]["size"])), 3),
                "research_rate": round(research[k]["bought"] / max(1, research[k]["offers"]), 3),
                "fees_paid_per_match": mean(k, "fees_paid"), "fees_received_per_match": mean(k, "fees_received"),
                "trades_per_match": mean(k, "trades"), "volume_per_match": mean(k, "volume"),
                "wrong_direction_per_match": mean(k, "wrong_direction"),
                "pass_when_should_trade_per_match": mean(k, "pass_when_should_trade"),
                "trade_when_should_pass_per_match": mean(k, "trade_when_should_pass"),
                "market_read": mean(k, "market_read"), "belief_logloss": mean(k, "belief_logloss"),
                "notes_written": notes[k],
            },
            "calls": len(calls[k]),
            "tokens_per_move_median": round(st.median(tok)) if tok else None,
            "tokens_per_move_mean": round(st.mean(tok)) if tok else None,
            "latency_s_median": round(st.median(lat), 1) if lat else None,
            "latency_s_p90": round(sorted(lat)[int(0.9 * len(lat))], 1) if lat else None,
            "cost_usd": round(cost, 2), "cost_usd_per_match": round(cost / max(1, seats), 3),
        }
        if k in elo:
            row["elo"], row["elo_ci95"] = round(elo[k]["elo"]), [round(elo[k]["lo"]), round(elo[k]["hi"])]
        out_models.append(row)
    return {
        "setup": {"models": len(models), "deals": len(deals), "matches": n_matches,
                  "seats": max((len(m["seating"]) for m in ms), default=0)},
        "deals": [{"unit": u, "deal": family_of[u][0], "seed": family_of[u][1], "family": family_of[u][2]}
                  for u in deals],
        "models": out_models,
        "cost": {"run_usd": round(sum(o["cost_usd"] for o in out_models), 2),
                 "match_wall_minutes_median": round(st.median(wall) / 60, 1) if wall else None},
    }
