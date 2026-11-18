"""Urn Exchange (game id ``exchange``). Determinism, zero-sum, the call auction, exact posteriors,
no leaks, duplicate slot evidence, legality, bots."""

from __future__ import annotations

import copy
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from urn_exchange import exchange as X
from urn_exchange.core import Bot, View, play_match
from urn_exchange.runner import default_registry

DEALS = X.Exchange().deals()
PILOTS = [d for d in DEALS if d.startswith("pilot")]
ALL_BOTS = sorted(X.BOTS)


def new(deal="pilot-01", n=3, seed=0) -> X.Exchange:
    g = X.Exchange()
    g.reset(deal, seed, n)
    return g


def order(side="pass", size=0, limit=None, p=0.5, **kw):
    a = {"side": side, "size": size, "belief": {"p_a": p}, **kw}
    if limit is not None:
        a["limit"] = limit
    return a


def research(flag=False, p=0.5, **kw):
    return {"research": flag, "belief": {"p_a": p}, **kw}


def bots(g, names):
    return [X.BOTS[b](g, s) for s, b in enumerate(names)]


def play(names, deal, seed=0):
    g = X.Exchange()
    return play_match(g, bots(g, names), deal, seed)


def snap(g: X.Exchange) -> str:
    return json.dumps({"hand": g.hand, "phase": g.phase, "pending": g.pending, "bankroll": g.bankroll,
                       "legs": g.legs, "researched": g.researched, "rounds": len(g.rounds),
                       "fees": [g.fees_paid, g.fees_received], "records": g.records, "cur": g.cur,
                       "views": [g.view(s).text for s in range(g.n_players)], "transcript": g.transcript},
                      sort_keys=True, default=str)


# --------------------------------------------------------------------------- registry / shape


def test_registered_and_deals():
    games, bots_ = default_registry()
    assert games["exchange"] is X.Exchange and X.GAME is X.Exchange
    cls = games["exchange"]
    assert (cls.kind, cls.min_players, cls.max_players, cls.version, cls.title) == ("multi", 3, 6, "0.1.0",
                                                                                      "Urn Exchange")
    assert set(bots_["exchange"]) >= {"random", "alwayspass", "bayes", "pooled", "follower"}
    assert DEALS == ["pilot-01", "pilot-02", "pilot-03", "pilot-04"] + [f"full-{i:02d}" for i in range(1, 13)]
    assert [X.deal_params(d)[0] for d in PILOTS] == ["clean", "research", "insider", "noisy"]
    assert {X.deal_params(d)[0] for d in DEALS if d.startswith("full")} == set(X.FAMILIES)
    assert list(X.HEADLINE) == ["chips", "belief_logloss", "market_read", "trade_regret"]
    assert X.HEADLINE["chips"] == "higher_is_better"


def test_hand_spec_is_a_function_of_deal_and_hand():
    for d in DEALS:
        fam, noise = X.deal_params(d)
        research, insider, noisy = X.FAMILIES[fam]
        for h in range(X.HANDS):
            s = X.hand_spec(d, h)
            assert s == X.hand_spec(d, h)
            assert (s.fee > 0) == research and (s.fee in X.FEES or s.fee == 0)
            assert (s.noise > 0) == noisy
            assert s.k_priv == 1 if insider else s.k_priv in (1, 2)
        for n in (3, 4, 5, 6):
            ins = X.insider_slot(d, n)
            assert (ins is not None) == insider and (ins is None or 0 <= ins < n)


# --------------------------------------------------------------------------- determinism

SCRIPT = r"""
from urn_exchange import exchange as X
from urn_exchange.core import play_match
out = []
for deal in ("pilot-02", "pilot-04", "full-05"):
    names = ["random", "bayes", "pooled", "follower"]
    g = X.Exchange()
    log = play_match(g, [X.BOTS[b](g, s) for s, b in enumerate(names)], deal, 3)
    out.append(log.result.to_dict())
    out.append([s["view"] for s in log.steps])
import json; print(json.dumps(out, sort_keys=True))
"""


def test_deterministic_across_runs_and_hash_seeds():
    outs = set()
    for hs in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        outs.add(subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True,
                                env=env).stdout)
    assert len(outs) == 1
    a = play(["random", "bayes", "pooled"], "full-06", 2).result.to_dict()
    b = play(["random", "bayes", "pooled"], "full-06", 2).result.to_dict()
    assert a == b
    c = play(["random", "bayes", "pooled"], "full-06", 3).result.to_dict()
    assert a["info"]["hands"] != c["info"]["hands"]  # the seed changes the world


# --------------------------------------------------------------------------- zero-sum and fees


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_zero_sum_with_fees(n):
    mixes = [["random"] * n, (["bayes", "random", "pooled", "follower", "noisy"])[:n],
             (["random", "bayes", "random", "halfkelly", "random"])[:n]]
    for d in DEALS:
        for seed in (0, 1):
            for names in mixes:
                log = play(names, d, seed)
                r = log.result
                assert sum(r.payoffs) == 0 and all(float(x).is_integer() for x in r.payoffs)
                assert sum(m["fees_paid"] for m in r.metrics) == sum(m["fees_received"] for m in r.metrics)
                assert all(m["final_chips"] >= 1 for m in r.metrics)
                for h in r.info["hands"]:
                    assert sum(h["pnl"]) == 0
                    for rd in h["rounds"]:
                        assert sum(rd["fills"].values()) == 0
                        assert sum(v for v in rd["fills"].values() if v > 0) == rd["volume"]


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_fee_split_is_whole_chips_and_zero_sum(n):
    g = new("pilot-02", n)
    seen_extra: Counter = Counter()
    for h in range(X.HANDS):
        g.hand, g.spec = h, X.hand_spec("pilot-02", h)
        for payer in range(n):
            sh = g.fee_shares(payer)
            assert payer not in sh and set(sh) == set(range(n)) - {payer}
            assert sum(sh.values()) == g.spec.fee
            base = g.spec.fee // (n - 1)
            assert set(sh.values()) <= {base, base + 1}
            assert sh == g.fee_shares(payer)  # deterministic
            seen_extra.update(s for s, c in sh.items() if c > base)
    # remainders are not always given to the same seat
    assert len(seen_extra) > 1


def test_research_pays_the_others_and_shows_one_more_ball():
    g = new("pilot-02", 4)
    fee = g.spec.fee
    assert g.phase == "research"
    before = list(g.bankroll)
    for s in range(4):
        assert g.check(s, research(s == 0)) is None
    sh = g.fee_shares(0)
    assert g.bankroll[0] == before[0] - fee
    assert all(g.bankroll[s] == before[s] + sh[s] for s in (1, 2, 3))
    assert g.researched == [True, False, False, False] and g.phase == "r1"
    assert len(g.own_reports(0)) == 1 + g.n_private(0) + 1 and len(g.own_reports(1)) == 1 + g.n_private(1)
    assert "1 of 4 players bought research" in g.view(1).text
    assert f"you received {sh[1]} chips" in g.view(1).text
    assert "the last is your research ball" in g.view(0).text


# --------------------------------------------------------------------------- the call auction


def brute_clear(orders):
    rows = []
    for p in range(1, 100):
        d = sum(n for _, sd, n, lim in orders if sd == "buy" and lim >= p)
        s = sum(n for _, sd, n, lim in orders if sd == "sell" and lim <= p)
        rows.append((p, min(d, s), abs(d - s)))
    vmax = max(v for _, v, _ in rows)
    if vmax == 0:
        return None, 0, []
    imb = min(i for _, v, i in rows if v == vmax)
    tied = [p for p, v, i in rows if v == vmax and i == imb]
    return tied, vmax, rows


def random_book(r: random.Random, n: int):
    orders = []
    for s in range(n):
        side = r.choice(["buy", "sell", "pass"])
        if side != "pass":
            centre = r.choice([30, 50, 70])
            orders.append((s, side, r.randint(1, 10), max(1, min(99, centre + r.randint(-25, 25)))))
    return orders


def test_auction_clears_like_brute_force_on_random_books():
    r = random.Random(7)
    crossed = 0
    for _ in range(3000):
        n = r.randint(3, 5)
        orders = random_book(r, n)
        price, vol = X.clearing_price(orders)
        tied, vmax, _ = brute_clear(orders)
        if tied is None:
            assert price is None and vol == 0
            bids = [lim for _, sd, _, lim in orders if sd == "buy"]
            asks = [lim for _, sd, _, lim in orders if sd == "sell"]
            assert not bids or not asks or max(bids) < min(asks)
            continue
        crossed += 1
        assert vol == vmax and price in tied
        assert tied == list(range(min(tied), max(tied) + 1))  # the tied set is an interval
        mid = (min(tied) + max(tied)) / 2
        assert abs(price - mid) <= 0.5
        if abs(price - mid) == 0.5:
            assert abs(price - 50) < abs((2 * mid - price) - 50)
        # mirror symmetry: p -> 100 - p with buy <-> sell
        mirror = [(s, "sell" if sd == "buy" else "buy", q, 100 - lim) for s, sd, q, lim in orders]
        assert X.clearing_price(mirror) == (100 - price, vol)
        # allocation
        tb = list(range(n))
        r.shuffle(tb)
        fills = X.allocate(orders, price, vol, tb)
        assert sum(q for q in fills.values() if q > 0) == vol == -sum(q for q in fills.values() if q < 0)
        by = {s: (sd, q, lim) for s, sd, q, lim in orders}
        for s, f in fills.items():
            sd, q, lim = by[s]
            assert (f > 0) == (sd == "buy") and abs(f) <= q
            assert (lim >= price) if sd == "buy" else (lim <= price)
        for side in ("buy", "sell"):
            elig = [(s, q, lim) for s, (sd, q, lim) in by.items() if sd == side
                    and (lim >= price if side == "buy" else lim <= price)]
            tot = sum(q for _, q, _ in elig)
            if tot == vol:
                assert all(abs(fills.get(s, 0)) == q for s, q, _ in elig)
                continue
            better = (lambda a, b: a > b) if side == "buy" else (lambda a, b: a < b)
            for (s1, q1, l1), (s2, q2, l2) in itertools.permutations(elig, 2):
                if better(l1, l2) and abs(fills.get(s2, 0)) > 0:
                    assert abs(fills.get(s1, 0)) == q1  # price priority
            marg = [(s, q, lim) for s, q, lim in elig if 0 < abs(fills.get(s, 0)) < q]
            if len(marg) > 1:
                lv = {lim for _, _, lim in marg}
                assert len(lv) == 1  # only the marginal level is rationed
                lim = lv.pop()
                level = [(s, q) for s, q, l2 in elig if l2 == lim]
                lt = sum(q for _, q in level)
                rem = sum(abs(fills.get(s, 0)) for s, _ in level)
                for s, q in level:  # pro rata within one contract
                    assert abs(abs(fills.get(s, 0)) - rem * q / lt) < 1
    assert crossed > 1000


def test_auction_examples_and_seat_symmetric_rationing():
    # one buyer 5 @60, one seller 5 @40: every price 40..60 matches 5 with zero imbalance -> middle, 50
    assert X.clearing_price([(0, "buy", 5, 60), (1, "sell", 5, 40)]) == (50, 5)
    assert X.clearing_price([(0, "buy", 5, 40), (1, "sell", 5, 60)]) == (None, 0)
    assert X.clearing_price([(0, "buy", 5, 60)]) == (None, 0)
    # tie 49..52: middle 50.5 -> nearer 50
    assert X.clearing_price([(0, "buy", 3, 52), (1, "sell", 3, 49)]) == (50, 3)
    assert X.clearing_price([(0, "buy", 3, 51), (1, "sell", 3, 48)]) == (50, 3)  # mid 49.5 -> 50
    # two equal buyers, one unit of supply: the deal-stream order decides, not the seat index
    orders = [(0, "buy", 1, 60), (1, "buy", 1, 60), (2, "sell", 1, 40)]
    p, v = X.clearing_price(orders)
    assert X.allocate(orders, p, v, [1, 0, 2]) == {1: 1, 2: -1}
    assert X.allocate(orders, p, v, [0, 1, 2]) == {0: 1, 2: -1}
    winners = Counter()
    g = new("pilot-01", 3)
    for h in range(X.HANDS):
        for seed in range(30):
            g.seed, g.hand = seed, h
            winners[g.tiebreak(1)[0]] += 1
    assert set(winners) == {0, 1, 2}


# --------------------------------------------------------------------------- exact posterior


def enum_posterior(spec: X.ExSpec, reports: list[bool]) -> float:
    """Brute force over the urn, every true colour and every misreport flip."""
    tot = {"A": 0.0, "B": 0.0}
    k = len(reports)
    for urn, pu in (("A", spec.prior), ("B", 1 - spec.prior)):
        red = spec.red_a if urn == "A" else spec.red_b
        for colours in itertools.product((True, False), repeat=k):
            for flips in itertools.product((True, False), repeat=k):
                rep = [c != f for c, f in zip(colours, flips)]
                if rep != reports:
                    continue
                pr = pu
                for c, f in zip(colours, flips):
                    pr *= (red if c else 1 - red) * (spec.noise if f else 1 - spec.noise)
                tot[urn] += pr
    return tot["A"] / (tot["A"] + tot["B"])


def test_posterior_matches_brute_force_enumeration():
    for d in DEALS:
        for h in (0, 3, 7):
            s = X.hand_spec(d, h)
            for k in range(1, 6):
                for reports in itertools.product((True, False), repeat=k):
                    assert X.own_posterior(s, list(reports)) == pytest.approx(enum_posterior(s, list(reports)),
                                                                              abs=1e-12)


def test_game_posteriors_read_the_right_balls():
    for d in ("pilot-03", "pilot-04", "full-05"):
        g = new(d, 5, 2)
        for s in range(5):
            rep = [g.public] + g.balls[s][:g.n_private(s)]
            assert g.own_reports(s) == rep
            assert g.own_post(s) == X.own_posterior(g.spec, rep)
        pooled = [g.public] + [b for s in range(5) for b in g.balls[s][:g.n_private(s)]]
        assert g.pooled_post() == pytest.approx(X.own_posterior(g.spec, pooled))
        if X.FAMILIES[g.family][1]:
            assert g.n_private(g.insider) == 3 and all(g.n_private(s) == 1 for s in range(5) if s != g.insider)


def test_posterior_is_calibrated_by_monte_carlo():
    """Group (deal, seed) worlds by the stated own posterior; the urn frequency must match."""
    bins: dict[int, list[int]] = defaultdict(list)
    for seed in range(400):
        g = new("pilot-04", 3, seed)
        for s in range(3):
            p = g.own_post(s)
            bins[int(p * 10)].append(1 if g.urn == "A" else 0)
    for b, xs in bins.items():
        if len(xs) >= 150:
            assert abs(statistics.mean(xs) - (b + 0.5) / 10) < 0.12


# --------------------------------------------------------------------------- Kelly maths


def test_best_at_price_matches_brute_force_and_bankroll_rule():
    r = random.Random(3)
    for _ in range(400):
        B = r.randint(50, 3000)
        legs = [(r.randint(-5, 5), r.randint(1, 99))] if r.random() < 0.5 else []
        if B + X.worst_case(legs) < 1:
            continue
        p, price = r.random(), r.randint(1, 99)
        best = max([(X.log_value(p, B, legs), 0)]
                   + [(X.log_value(p, B, legs + [(q, price)]), q) for q in range(-10, 11)
                      if q and X.legal(B, legs, "buy" if q > 0 else "sell", abs(q), price)],
                   key=lambda t: t[0])
        v, q = X.best_at_price(p, B, legs, price)
        assert v == pytest.approx(best[0], abs=1e-12)
        for side in ("buy", "sell"):
            m = X.legal_max(B, legs, side, price)
            assert all(X.legal(B, legs, side, k, price) for k in range(1, m + 1))
            assert m == 10 or not X.legal(B, legs, side, m + 1, price)


def test_bankroll_rule_counts_the_round_one_position():
    g = new("pilot-01", 3)
    g.bankroll[0] = 200
    for s in range(3):
        assert g.check(s, [order("buy", 3, 60), order("sell", 3, 40), order()][s]) is None
    assert g.legs[0] == [(3, 50)] and g.phase == "r2"
    # worst case with +3 @50 already is -150 if B: one more buy at 99 risks 99 each -> at most 0
    err = g.check(0, order("buy", 1, 99))
    assert err and "the most you can buy at limit 99 is 0" in err
    assert g.check(0, order("sell", 3, 50)) is None  # closing the position is always allowed


# --------------------------------------------------------------------------- settlement


def test_settlement_and_views_after_the_hand():
    g = new("pilot-01", 3)
    urn = g.urn
    for s in range(3):
        assert g.check(s, [order("buy", 5, 60), order("sell", 5, 40), order()][s]) is None
    assert g.rounds[0]["price"] == 50 and g.rounds[0]["fills"] == {0: 5, 1: -5}
    assert "Round 1: cleared at 50, 5 contracts. You bought 5 @50." in g.view(0).text
    assert "Round 1: cleared at 50, 5 contracts. Your fill: none." in g.view(2).text
    for s in range(3):
        assert g.check(s, order()) is None
    win = 250 if urn == "A" else -250
    assert g.bankroll == [1000 + win, 1000 - win, 1000] and g.hand == 1
    v = g.view(0).text
    assert v.startswith(f"Last hand (1): urn {urn}. Prices: round 1 50 x 5; round 2 no trade. You bought 5 @50: "
                        f"{win:+d}.")


# --------------------------------------------------------------------------- no leaks


def _seat0_views(deal, n, patch=None, seed=0):
    """Every view seat 0 gets with random bots elsewhere (their actions never read balls)."""
    g = X.Exchange()
    players = [X.BOTS["alwayspass"](g, 0)] + [X.BOTS["random"](g, s) for s in range(1, n)]
    orig = X.ball_report
    if patch:
        X.ball_report = patch(orig)
    try:
        log = play_match(g, players, deal, seed)
    finally:
        X.ball_report = orig
    return [s["view"] for s in log.steps if s["seat"] == 0 and s["view"]], g


@pytest.mark.parametrize("deal", PILOTS + ["full-05", "full-06"])
@pytest.mark.parametrize("n", [3, 5])
def test_no_leak_of_other_seats_balls(deal, n):
    def flip_others(orig):
        def f(deal_id, seed, hand, slot, j, red, noise):
            x = orig(deal_id, seed, hand, slot, j, red, noise)
            return (not x) if slot not in ("pub", 0) else x
        return f

    base, g1 = _seat0_views(deal, n)
    other, g2 = _seat0_views(deal, n, flip_others)
    assert [h["balls"][1:] for h in g1.result().info["hands"]] != [h["balls"][1:] for h in g2.result().info["hands"]]
    assert base == other


def test_urn_hidden_until_the_hand_settles():
    for deal in PILOTS:
        for seed in range(5):
            g = new(deal, 4, seed)
            views = []
            while g.hand == 0:
                for s in g.to_act():
                    before = [g.view(t).text for t in range(4)]
                    g.urn = "B" if g.urn == "A" else "A"  # the urn alone (balls fixed) never shows
                    assert [g.view(t).text for t in range(4)] == before
                    g.urn = "B" if g.urn == "A" else "A"
                    views.append(g.view(s).text)
                    assert g.check(s, g.default_action(s)) is None
            assert not any("urn A." in v or "urn B." in v for v in views)


def test_views_never_render_balls_as_letters():
    for deal in PILOTS:
        g = X.Exchange()
        log = play_match(g, bots(g, ["bayes", "random", "pooled", "follower"]), deal)
        for st in log.steps:
            if st["view"]:
                for line in st["view"].splitlines():
                    if "ball" in line and ":" in line:
                        assert " R," not in line and " B," not in line and not line.rstrip(".").endswith((" R", " B"))
                assert len(st["view"]) <= 700
        for s in range(4):
            assert len(g.rules(s)) <= 3500


# --------------------------------------------------------------------------- duplicate slot evidence


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_rotation_gives_every_model_identical_slot_evidence(n):
    mix = ["bayes", "random", "pooled", "follower", "alwayspass", "halfkelly"][:n]
    for deal in ("pilot-03", "full-05"):
        seen = []
        insider_by_bot = Counter()
        for rot in range(n):
            names = mix[rot:] + mix[:rot]
            r = play(names, deal, 1).result
            spec = [(h["urn"], h["public"], [b[:X.hand_spec(deal, i).k_priv if s != r.info["insider"] else 3]
                                             for s, b in enumerate(h["balls"])])
                    for i, h in enumerate(r.info["hands"])]
            seen.append(spec)
            if r.info["insider"] is not None:
                insider_by_bot[names[r.info["insider"]]] += 1
        assert all(x == seen[0] for x in seen)
        if insider_by_bot:
            assert insider_by_bot == Counter({b: 1 for b in mix})
    # the slot's balls do not depend on the table size either
    for slot in range(3):
        assert {tuple(new("pilot-01", n, 4).balls[slot][:2]) for n in (3, 4, 5, 6)} == \
            {tuple(new("pilot-01", 3, 4).balls[slot][:2])}


# --------------------------------------------------------------------------- legality


BAD_RESEARCH = [
    {"research": "yes", "belief": {"p_a": 0.5}}, {"research": True}, research(True, 1.5), research(True, -0.1),
    research(False, float("nan")), {"research": False, "belief": {}}, {"research": False, "belief": {"p_a": True}},
    {"research": False, "belief": {"p_a": 0.5, "x": 1}}, research(False, note="x" * 201),
    {"side": "pass", "size": 0, "belief": {"p_a": 0.5}}, [], "research",
]
BAD_ORDERS = [
    order("buy", 11, 50), order("sell", 0, 50), order("buy", -1, 50), order("buy", 2.5, 50), order("pass", 3),
    order("hold", 1, 50), order("buy", 1), order("buy", 1, 0), order("buy", 1, 100), order("buy", 1, 50.5),
    order("buy", 1, 50, 1.5), order("pass", 0, p="high"), {"side": "pass", "size": 0},
    {"side": "pass", "size": 0, "belief": {"p_a": 0.5}, "research": True}, order("pass", note="x" * 201),
    order("sell", True, 50), [], "buy",
]


@pytest.mark.parametrize("action", BAD_RESEARCH)
def test_illegal_research_actions_leave_state_unchanged(action):
    g = new("pilot-02", 3)
    before = snap(g)
    assert g.check(0, action) is not None
    assert snap(g) == before


@pytest.mark.parametrize("action", BAD_ORDERS)
def test_illegal_orders_leave_state_unchanged(action):
    g = new("pilot-01", 3)
    before = snap(g)
    assert g.check(0, action) is not None
    assert snap(g) == before


def test_bankroll_and_turn_errors():
    g = new("pilot-02", 3)
    g.bankroll[0] = g.spec.fee  # cannot keep 1 chip after the fee
    before = snap(g)
    assert "cannot afford" in g.check(0, research(True))
    assert snap(g) == before
    assert g.check(0, research(False)) is None
    assert g.check(0, research(False)) == "seat 0 is not due to act"  # already acted this phase
    g = new("pilot-01", 3)
    g.bankroll[1] = 300
    err = g.check(1, order("buy", 10, 40))
    assert err and "the most you can buy at limit 40 is 7" in err
    assert g.check(1, order("buy", 7, 40)) is None
    assert g.check(1, order("sell", 1, 99)) == "seat 1 is not due to act"
    with pytest.raises(KeyError):
        g.reset("pilot-09", 0, 3)
    with pytest.raises(ValueError):
        g.reset("pilot-01", 0, 2)
    with pytest.raises(ValueError):
        g.reset("pilot-01", 0, 7)


def test_default_action_legal_in_every_phase_and_always_invalid_player_completes():
    for deal in ("pilot-02", "full-05"):
        g = new(deal, 4)
        while not g.done:
            for s in g.to_act():
                cp = copy.deepcopy(g)
                assert cp.check(s, cp.default_action(s)) is None
                assert g.check(s, X.BOTS["random"](g, s).act(g.view(s), g.action_schema(s))) is None
        g = X.Exchange()
        junk = Bot("junk", g, 0, lambda gg, ss, v: {"side": "buy", "size": 99, "belief": {"p_a": 0.5}})
        log = play_match(g, [junk] + bots(g, ["bayes", "bayes", "random"])[1:], deal)
        assert log.invalid[0] > 0
        assert log.defaulted[0] == sum(1 for st in log.steps if st["defaulted"] and st["seat"] == 0) > 0
        assert log.result.metrics[0]["volume"] == 0 and sum(log.result.payoffs) == 0


def test_errors_name_the_problem():
    g = new("pilot-01", 3)
    assert "needs a limit" in g.check(0, order("buy", 2))
    assert "size must be 0 for pass" in g.check(0, order("pass", 2))
    assert "use pass" in g.check(0, order("sell", 0, 50))


# --------------------------------------------------------------------------- full matches and bots


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_full_matches_with_every_bot_complete_and_serialise(n):
    for i, deal in enumerate(DEALS):
        names = [ALL_BOTS[(i + j) % len(ALL_BOTS)] for j in range(n)]
        log = play(names, deal, i)
        r = log.result
        assert sum(log.invalid) == 0 and sum(log.defaulted) == 0
        assert sum(r.payoffs) == 0 and len(r.payoffs) == n and len(r.metrics) == n
        json.dumps(r.to_dict(), allow_nan=False)
        research = X.FAMILIES[X.deal_params(deal)[0]][0]
        for m in r.metrics:
            assert m["hands"] == X.HANDS and m["decisions"] == X.HANDS * (3 if research else 2)
            assert m["trade_regret"] >= 0 and m["market_read"] >= 0 and m["belief_logloss"] >= 0
            assert m["trade_regret"] == pytest.approx(m["trade_regret_r1"] + m["trade_regret_r2"])
        assert len(r.info["hands"]) == X.HANDS
        assert sum(m["insider"] for m in r.metrics) == (1 if X.FAMILIES[X.deal_params(deal)[0]][1] else 0)


def _duplicate_chips(mix, deals, seeds):
    chips: dict[str, list[float]] = defaultdict(list)
    mets: dict[str, list[dict]] = defaultdict(list)
    n = len(mix)
    for d in deals:
        for seed in seeds:
            tot = defaultdict(float)
            for rot in range(n):
                names = mix[rot:] + mix[:rot]
                r = play(names, d, seed).result
                for s, b in enumerate(names):
                    tot[b] += r.payoffs[s]
                    mets[b].append(r.metrics[s])
            for b, v in tot.items():
                chips[b].append(v)
    return chips, mets


def test_bot_ordering_on_chips_over_many_seeds():
    mix = ["pooled", "bayes", "random", "follower", "alwayspass"]
    chips, mets = _duplicate_chips(mix, DEALS, range(3))
    mean = {b: statistics.mean(v) for b, v in chips.items()}
    assert mean["pooled"] > mean["bayes"] > mean["random"]
    assert mean["pooled"] > 0 > mean["random"]
    wins = sum(p > q for p, q in zip(chips["bayes"], chips["random"]))
    assert wins > 0.75 * len(chips["bayes"])
    avg = lambda b, k: statistics.mean(m[k] for m in mets[b])  # noqa: E731
    assert avg("bayes", "belief_logloss") < 1e-6 and avg("pooled", "market_read") < 1e-3  # 0 up to the 0.001 clip
    assert avg("follower", "market_gain") > 0 > avg("random", "market_gain")
    assert avg("random", "belief_logloss") > avg("alwayspass", "belief_logloss") > 0
    assert all(m["volume"] == 0 and m["draws"] == 0 for m in mets["alwayspass"])


def test_bayes_beats_random_heads_up_at_three_players():
    chips, _ = _duplicate_chips(["bayes", "random", "random"], DEALS, range(2))
    assert statistics.mean(chips["bayes"]) > 0 > statistics.mean(chips["random"])


def test_bayes_research_is_sometimes_worth_it_and_sometimes_not():
    decisions = Counter()
    for seed in range(10):
        g = new("pilot-02", 3, seed)
        for h in range(X.HANDS):
            g.hand, g.spec = h, X.hand_spec("pilot-02", h)
            decisions[X.research_worth_it(g, 0, g.spec.prior)] += 1
    assert decisions[True] > 0 and decisions[False] > 0


def test_bot_orders_are_always_legal():
    for deal in DEALS:
        g = X.Exchange()
        log = play_match(g, bots(g, ["noisy", "halfkelly", "pooled", "bayes", "follower"]), deal, 5)
        assert sum(log.invalid) == 0


def test_view_and_rules_mention_the_essentials():
    g = new("pilot-03", 4)
    r = g.rules(0)
    for s in ("zero-sum", "ONE price", "pro rata", "BANKROLL RULE", "Kelly", "insider",
              "not investment advice", "ALL players' balls", "red or blue"):
        assert s in r
    assert f"S{g.insider + 1}" in r
    r2 = new("pilot-02", 4).rules(1)
    assert "split equally among the other 3 players" in r2 and "remainder" in r2
    v = g.view(0)
    assert isinstance(v, View) and "round 1 of 2" in v.text and "public ball" in v.text
