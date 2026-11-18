"""Urn Exchange power analysis: how many deals separate two players on chips? Offline, bots only, free.

Urn Exchange is ranked by chips (duplicate play), so there must be enough deals for luck to wash out.
One DEAL here = one (deal id, seed) block played by a 5-seat table in all 5 rotations (5 matches x 10 hands),
exactly the unit the ratings use (``urn_exchange.ratings.multi_units``).

Method:
1. Simulate a pool of blocks per table (the 12 full deals x seeds 0..S-1, so the family mix is the paid
   config's), recording each bot's chips summed over the 5 rotations.
2. For each number of deals N in the grid, draw ``--reps`` random sets of N blocks from the pool (with
   replacement) and test every listed pair two ways:
   * paired bootstrap on chips: the 95% percentile CI of the mean per-deal chip difference excludes 0;
   * BT: the duplicate-group Bradley-Terry fit of the ratings (same prior, same comparisons: each pair of
     bots in a block compared on summed chips, win/draw/loss) with a bootstrap over blocks; "separated" =
     the 95% CI of the Elo DIFFERENCE excludes 0 (``bt_diff``), and, stricter, the two players' own 95%
     Elo intervals do not overlap (``bt_disjoint``, what a reader of the report table sees).
   Power = share of the reps that separate. Reported: the smallest N with power >= 80%. The ``null`` table seats
   two copies of bayes; how often they "separate" is each test's false-positive rate at that N (percentile
   bootstraps are anti-conservative with few deals, so read power only where the false-positive rate is ~5%).
3. Normal approximation for pairs that do not get there inside the grid: N80 ~ ((1.96 + 0.84) sd / mean)^2
   from the pool's per-deal chip differences.

Bots stand in for models of different strength: pooled (sees every seat's balls), bayes (exact own-information
posterior, Kelly order, ignores the price), halfkelly (bayes at half size), noisy03 / noisy06 (bayes with log-odds
belief noise of SD 0.3 / 0.6 per decision), follower (bayes in round 1, passes in round 2), random, alwayspass.

Run (about 2-4 minutes on 8 cores): python scripts/power.py
Research only, paper chips, not investment advice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from urn_exchange import exchange as X
from urn_exchange.core import play_match
from urn_exchange.ratings import ELO_SCALE, PRIOR_SD_ELO, bt_fit

FULL = [f"full-{i:02d}" for i in range(1, 13)]

TABLES = {
    # wide spread of strength: oracle, exact, half-informed, noise, nothing
    "spread": ["pooled", "bayes", "follower", "random", "alwayspass"],
    # close-in-skill strategic players, plus one random seat for liquidity (models will not all be sharp)
    "close": ["bayes", "noisy03", "noisy06", "halfkelly", "random"],
    # calibration: two copies of the same bot, so every "separation" of bayes vs bayes2 is a false positive
    "null": ["bayes", "bayes2", "noisy03", "halfkelly", "random"],
}
PAIRS = {
    "spread": [("pooled", "bayes"), ("bayes", "random"), ("bayes", "follower"), ("bayes", "alwayspass")],
    "close": [("bayes", "random"), ("bayes", "noisy06"), ("bayes", "halfkelly"), ("bayes", "noisy03"),
              ("noisy03", "noisy06")],
    "null": [("bayes", "bayes2")],
}
BOT_FACTORIES = {
    **X.BOTS,
    "noisy03": X.make_bayes("noisy03", noise_sd=0.3),
    "noisy06": X.make_bayes("noisy06", noise_sd=0.6),
    "bayes2": X.make_bayes("bayes2"),
}
GRID = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


# --------------------------------------------------------------------------- simulation


def play_block(args: tuple[str, tuple[str, ...], str, int]) -> dict[str, float]:
    """Chips per bot summed over the 5 rotations of one (deal, seed) block."""
    _, mix, deal, seed = args
    n = len(mix)
    tot = {b: 0.0 for b in mix}
    g = X.Exchange()
    for rot in range(n):
        names = mix[rot:] + mix[:rot]
        r = play_match(g, [BOT_FACTORIES[b](g, s) for s, b in enumerate(names)], deal, seed).result
        assert sum(r.payoffs) == 0
        for s, b in enumerate(names):
            tot[b] += r.payoffs[s]
    return tot


def simulate(table: str, seeds: int, workers: int) -> np.ndarray:
    mix = tuple(TABLES[table])
    jobs = [(table, mix, d, s) for s in range(seeds) for d in FULL]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(play_block, jobs, chunksize=4))
    return np.array([[r[b] for b in mix] for r in rows])  # (blocks, players)


# --------------------------------------------------------------------------- BT, vectorised over bootstrap draws


def block_scores(chips: np.ndarray) -> np.ndarray:
    """(blocks, k, k): score of i vs j in each block on summed chips (1 win, 0.5 draw, 0 loss); diag 0."""
    d = chips[:, :, None] - chips[:, None, :]
    s = np.where(np.abs(d) <= 1e-9, 0.5, (d > 0).astype(float))
    k = chips.shape[1]
    s[:, np.arange(k), np.arange(k)] = 0.0
    return s


def bt_batch(W: np.ndarray, C: np.ndarray, iters: int = 40) -> np.ndarray:
    """Penalised BT (same prior as ``ratings.bt_fit``) for a batch: W (B,k,k) summed scores, C (B,k,k) counts.
    Returns Elo (B,k), mean 1500."""
    B, k, _ = W.shape
    lam = (ELO_SCALE / PRIOR_SD_ELO) ** 2
    th = np.zeros((B, k))
    diag = np.arange(k)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(th[:, :, None] - th[:, None, :])))
        g = (W - C * p).sum(axis=2) - lam * th
        H = C * p * (1 - p)  # off-diagonal; C has a zero diagonal
        H[:, diag, diag] = -H.sum(axis=2) - lam
        step = np.linalg.solve(H, g[..., None])[..., 0]
        th = th - step
        if np.max(np.abs(step)) < 1e-10:
            break
    th = th - th.mean(axis=1, keepdims=True)
    return 1500.0 + ELO_SCALE * th


def _check_bt(scores: np.ndarray, names: list[str]) -> None:
    """The vectorised fit must equal urn_exchange.ratings.bt_fit on the same comparisons."""
    k = len(names)
    comps = [(names[i], names[j], float(s[i, j]), 1.0) for s in scores[:7] for i, j in itertools.combinations(range(k), 2)]
    ref = bt_fit(names, comps)
    W = scores[:7].sum(0)
    C = np.ones((k, k)) * 7
    np.fill_diagonal(C, 0)
    got = bt_batch(W[None], C[None])[0]
    assert np.allclose(got, [ref[n] for n in names], atol=1e-6), (got, ref)


# --------------------------------------------------------------------------- power


def power_table(chips: np.ndarray, names: list[str], pairs: list[tuple[str, str]], reps: int, boot: int,
                rng: np.random.Generator) -> dict:
    k = len(names)
    idx = {n: i for i, n in enumerate(names)}
    scores = block_scores(chips)
    _check_bt(scores, names)
    C1 = np.ones((k, k))
    np.fill_diagonal(C1, 0)
    out = {f"{a}>{b}": {"chips": {}, "bt_diff": {}, "bt_disjoint": {}} for a, b in pairs}
    P = len(chips)
    for N in GRID:
        hits = {key: np.zeros(3) for key in out}
        for _ in range(reps):
            pick = rng.integers(0, P, N)
            ch, sc = chips[pick], scores[pick]
            bidx = rng.integers(0, N, (boot, N))  # bootstrap over the N deals
            elo_bs = bt_batch(sc[bidx].sum(1), np.broadcast_to(C1 * N, (boot, k, k)))
            lo, hi = np.percentile(elo_bs, [2.5, 97.5], axis=0)
            for a, b in pairs:
                i, j = idx[a], idx[b]
                d = ch[:, i] - ch[:, j]
                m = d[bidx].mean(1)
                cl, chh = np.percentile(m, [2.5, 97.5])
                dd = elo_bs[:, i] - elo_bs[:, j]
                dl, dh = np.percentile(dd, [2.5, 97.5])
                hits[f"{a}>{b}"] += [cl > 0 or chh < 0, dl > 0 or dh < 0, (lo[i] > hi[j]) or (lo[j] > hi[i])]
        for key in out:
            out[key]["chips"][N], out[key]["bt_diff"][N], out[key]["bt_disjoint"][N] = (hits[key] / reps).tolist()
    for a, b in pairs:
        key = f"{a}>{b}"
        d = chips[:, idx[a]] - chips[:, idx[b]]
        mu, sd = float(d.mean()), float(d.std(ddof=1))
        out[key]["mean_diff"] = mu
        out[key]["sd_diff"] = sd
        out[key]["win_rate"] = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
        out[key]["n80_normal"] = math.ceil(((1.96 + 0.84) * sd / abs(mu)) ** 2) if mu else None
        for m in ("chips", "bt_diff", "bt_disjoint"):
            ok = [N for N in GRID if out[key][m][N] >= 0.8]
            out[key][f"n80_{m}"] = ok[0] if ok else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=40, help="seeds per full deal in the pool (pool = 12 x seeds)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261118)
    res = {}
    for table in TABLES:
        chips = simulate(table, args.seeds, args.workers)
        t1 = time.time()
        res[table] = {"players": TABLES[table], "blocks": len(chips),
                      "mean_chips_per_deal": dict(zip(TABLES[table], chips.mean(0).round(1).tolist())),
                      "pairs": power_table(chips, TABLES[table], PAIRS[table], args.reps, args.boot, rng)}
        print(f"# {table}: {len(chips)} blocks simulated, power in {time.time() - t1:.0f}s", flush=True)
    res["config"] = {"seeds": args.seeds, "reps": args.reps, "boot": args.boot, "grid": list(GRID),
                     "wall_s": round(time.time() - t0, 1)}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    print()
    print("| table | pair (A > B) | mean chips A-B per deal | SD | A wins deal | N80 chips bootstrap | N80 BT diff CI "
          "| N80 BT CIs disjoint | N80 normal approx |")
    print("|---|---|---|---|---|---|---|---|---|")
    fmt = lambda x: "> 128" if x is None else str(x)  # noqa: E731
    for table in TABLES:
        for key, r in res[table]["pairs"].items():
            print(f"| {table} | {key.replace('>', ' > ')} | {r['mean_diff']:.0f} | {r['sd_diff']:.0f} | "
                  f"{100 * r['win_rate']:.0f}% | {fmt(r['n80_chips'])} | {fmt(r['n80_bt_diff'])} | "
                  f"{fmt(r['n80_bt_disjoint'])} | {r['n80_normal']} |")
    print()
    for table in TABLES:
        print(f"{table}: mean chips per deal (5 rotations x 10 hands): {res[table]['mean_chips_per_deal']}")
    print()
    print("Power by N (chips bootstrap / BT diff / BT disjoint):")
    for table in TABLES:
        for key, r in res[table]["pairs"].items():
            row = " ".join(f"{N}:{r['chips'][N]:.2f}/{r['bt_diff'][N]:.2f}/{r['bt_disjoint'][N]:.2f}" for N in GRID)
            print(f"  {table} {key}: {row}")
    print(f"\nwall {res['config']['wall_s']}s")


if __name__ == "__main__":
    main()
