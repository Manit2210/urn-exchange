"""Exact urn maths shared with the solo game: posterior by brute force, optimal stopping, Kelly sizing."""

from __future__ import annotations

import itertools
import math

import pytest

from urn_exchange import urns as U


def spec_with(**kw):
    base = dict(prior=0.5, red_a=0.9, red_b=0.1, n0=0, max_extra=2, fee=3, noise=0.0, informed=0.0, half=2)
    base.update(kw)
    return U.HandSpec(**base)


def brute_force(spec: U.HandSpec, reports: tuple[bool, ...], offset: int) -> float:
    """Sum over the urn, every true colour, every misreport flip and (informed quotes) the dealer type."""
    mass = {}
    for urn, pu in (("A", spec.prior), ("B", 1 - spec.prior)):
        red = spec.red_a if urn == "A" else spec.red_b
        q = spec.informed
        if q <= 0:
            pq = 1.0 if offset == 0 else 0.0
        else:
            informed_offset = U.SHADE if urn == "A" else -U.SHADE
            pq = q * (offset == informed_offset) + (1 - q) / len(U.OFFSETS)
        pd = 0.0
        for colours in itertools.product((True, False), repeat=len(reports)):
            for flips in itertools.product((True, False), repeat=len(reports)):
                if [c != f for c, f in zip(colours, flips)] != list(reports):
                    continue
                pw = 1.0
                for c, f in zip(colours, flips):
                    pw *= (red if c else 1 - red) * (spec.noise if f else 1 - spec.noise)
                pd += pw
        mass[urn] = pu * pq * pd
    return mass["A"] / (mass["A"] + mass["B"])


@pytest.mark.parametrize("noise,informed", [(0.0, 0.0), (0.15, 0.0), (0.0, 0.4), (0.2, 0.5)])
def test_posterior_matches_brute_force(noise, informed):
    checked = 0
    for prior in U.PRIORS[:6]:
        for red_a, red_b in U.COMPOSITIONS[:6]:
            s = spec_with(prior=prior, red_a=red_a, red_b=red_b, noise=noise, informed=informed)
            for n in range(0, 5):
                for reps in itertools.product((True, False), repeat=n):
                    for off in ((0,) if informed <= 0 else U.OFFSETS):
                        reds = sum(reps)
                        assert U.posterior(s, reds, n - reds, off) == pytest.approx(brute_force(s, reps, off), abs=1e-12)
                        checked += 1
    assert checked > 500


def test_value_of_information_sometimes_worth_it_sometimes_not():
    s = spec_with()
    good = U.solve(s, 48, 52, 0.5, 1000, 2)
    assert good["action"] == "draw" and good["v_star"] > good["v_imm"]
    weak = U.solve(spec_with(red_a=0.55, red_b=0.45, fee=20), 48, 52, 0.5, 1000, 2)
    assert weak["action"] == "pass" and weak["v_star"] == weak["v_imm"]
    free = U.solve(spec_with(fee=0, red_a=0.55, red_b=0.45), 48, 52, 0.5, 1000, 2)
    assert free["q_draw"] >= free["v_imm"] - 1e-12  # information never hurts when it is free
    done = U.solve(s, 48, 52, 0.5, 1000, 0)
    assert done["q_draw"] is None and done["action"] != "draw"
    for p in (0.1, 0.4, 0.5, 0.9):
        r = U.solve(s, 48, 52, p, 900, 2)
        assert r["v_star"] >= r["v_imm"] - 1e-12


def test_optimal_stopping_matches_independent_recursion():
    """Re-derive the value of 'draw then play optimally' with a from-scratch expectimax over worlds."""
    s = spec_with(red_a=0.75, red_b=0.4, fee=4, noise=0.1, prior=0.45)
    bid, ask, B = 44, 48, 800

    def V(reds, blues, bank, left):
        p = U.posterior(s, reds, blues, 0)
        best = max([math.log(bank)] + [U.trade_value(p, bank, bid, ask, side, n)
                                       for side in ("buy", "sell") for n in range(1, U.max_size(side, bank, bid, ask) + 1)])
        if left == 0 or bank - s.fee < 1:
            return best
        pr_red = p * U.report_red_prob(s.red_a, s.noise) + (1 - p) * U.report_red_prob(s.red_b, s.noise)
        cont = pr_red * V(reds + 1, blues, bank - s.fee, left - 1) + (1 - pr_red) * V(reds, blues + 1, bank - s.fee, left - 1)
        return max(best, cont)

    got = U.solve(s, bid, ask, U.posterior(s, 0, 0, 0), B, 2)
    assert got["v_star"] == pytest.approx(V(0, 0, B, 2), abs=1e-12)


def test_log_sizing_is_interior_for_weak_edges_and_capped_for_strong_ones():
    v, side, n = U.best_terminal(0.56, 1000, 48, 52)
    assert side == "buy" and 1 <= n < U.MAX_SIZE
    v, side, n = U.best_terminal(0.95, 1000, 48, 52)
    assert side == "buy" and n == U.MAX_SIZE
    assert U.best_terminal(0.5, 1000, 48, 52)[1:] == ("pass", 0)
    assert U.best_terminal(0.1, 1000, 48, 52)[1] == "sell"
