# Urn Exchange ratings

Paper chips, research and fun only; not investment advice. Definitions: docs/harness.md (Ratings).

Bootstrap: 1000 resamples over units, 95% percentile intervals.

## exchange (multi)

Matches ok: 72; not ok: 0.

**chips** (total over all matches; 95% bootstrap range over duplicate deals; ev = luck-free chips per match)

| rank | model | chips | 95% range | deals > 0 | ev per match |
|---|---|---|---|---|---|
| 1 | gemini-4-carbon | +4,612 | 548 to 8839 | 10/12 | 111.8 |
| 2 | gemini-4-argon | +3,619 | 788 to 6528 | 9/12 | 11.6 |
| 3 | claude-sonnet-5.5 | -315 | -3806 to 3270 | 5/12 | -8.3 |
| 4 | claude-fable-5.2 | -1,635 | -4826 to 1541 | 5/12 | -48.0 |
| 5 | gpt-6.2-sol | -2,675 | -6235 to 916 | 3/12 | -54.8 |
| 6 | gpt-6.2-astra | -3,606 | -8411 to 1191 | 4/12 | -12.4 |

**bt** (12 units)

| rank | model | Elo | 95% CI | comparisons |
|---|---|---|---|---|
| 1 | gemini-4-argon | 1616 | 1519 to 1769 | 60 |
| 2 | gemini-4-carbon | 1605 | 1490 to 1746 | 60 |
| 3 | claude-sonnet-5.5 | 1520 | 1413 to 1643 | 60 |
| 4 | claude-fable-5.2 | 1459 | 1346 to 1561 | 60 |
| 5 | gpt-6.2-sol | 1406 | 1279 to 1510 | 60 |
| 6 | gpt-6.2-astra | 1395 | 1200 to 1541 | 60 |

## Skill profile

Mean of each game's headline metric; rank in brackets.

| model | exchange: belief_logloss | exchange: chips | exchange: market_read | exchange: trade_regret |
|---|---|---|---|---|
| claude-fable-5.2 | 2.78e-06 (3) | -22.7 (4) | 0.0909 (5) | 385 (5) |
| claude-sonnet-5.5 | 0.000342 (6) | -4.38 (3) | 0.129 (6) | 414 (6) |
| gemini-4-argon | 0.000144 (5) | 50.3 (2) | 0.0501 (3) | 364 (4) |
| gemini-4-carbon | 4.14e-09 (2) | 64.1 (1) | 0.0326 (1) | 360 (3) |
| gpt-6.2-astra | 8.25e-10 (1) | -50.1 (6) | 0.041 (2) | 290 (1) |
| gpt-6.2-sol | 3.55e-05 (4) | -37.2 (5) | 0.0556 (4) | 356 (2) |

## Highlights

- **exchange** `exchange/full-04/s0/g0/r4/claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra+gpt-6.2-sol+gemini-4-argon+gemini-4-carbon` payoffs [290.0, -594.0, -380.0, 546.0, 948.0, -810.0]; revealed: {"deal": "full-04", "family": "noisy", "hands": [{"balls": [["red", "red"], ["red", "blue"], ["blue", "blue"], ["red", "blue"], ["red", "blue"], ["blue", "blue"]], "bankroll": [961, 1000, 1039, 1000, 1000, 1000], "hand": 1, "pnl": [-39, 0, 39, 0, 0, 0], "pooled_posterior": 0.02322046350340246, "p...
  - seat 2 (gpt-6.2-astra): `{"belief": {"p_a": 0.076}, "limit": 9, "side": "sell", "size": 2}`
  - seat 3 (gpt-6.2-sol): `{"belief": {"p_a": 0.091}, "limit": 13, "side": "sell", "size": 6}`
  - seat 4 (gemini-4-argon): `{"belief": {"p_a": 0.08}, "limit": 12, "note": "Cleared at 15x1 -> likely 1-2 private Reds among others -> P(A)~0.08", "side": "sell", "size": 6}`
  - seat 5 (gemini-4-carbon): `{"belief": {"p_a": 0.134}, "limit": 14, "side": "sell", "size": 1}`
- **exchange** `exchange/full-04/s0/g0/r3/gemini-4-carbon+claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra+gpt-6.2-sol+gemini-4-argon` payoffs [183.0, -409.0, 57.0, 104.0, 789.0, -724.0]; revealed: {"deal": "full-04", "family": "noisy", "hands": [{"balls": [["red", "red"], ["red", "blue"], ["blue", "blue"], ["red", "blue"], ["red", "blue"], ["blue", "blue"]], "bankroll": [925, 1000, 1000, 1000, 1000, 1075], "hand": 1, "pnl": [-75, 0, 0, 0, 0, 75], "pooled_posterior": 0.02322046350340246, "p...
  - seat 2 (claude-sonnet-5.5): `{"belief": {"p_a": 0.13}, "limit": 50, "side": "pass", "size": 0}`
  - seat 3 (gpt-6.2-astra): `{"belief": {"p_a": 0.07}, "limit": 8, "side": "sell", "size": 5}`
  - seat 4 (gpt-6.2-sol): `{"belief": {"p_a": 0.087}, "limit": 12, "side": "sell", "size": 7}`
  - seat 5 (gemini-4-argon): `{"belief": {"p_a": 0.068}, "limit": 9, "note": "cleared 14 x 2, we bought all 2 -> others have B", "side": "sell", "size": 4}`
- **exchange** `exchange/full-06/s1/g0/r4/claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra+gpt-6.2-sol+gemini-4-argon+gemini-4-carbon` payoffs [1056.0, -95.0, 54.0, -167.0, -455.0, -393.0]; revealed: {"deal": "full-06", "family": "noisy_research", "hands": [{"balls": [["red", "red", "blue"], ["red", "blue"], ["red", "blue", "red"], ["blue", "blue", "red"], ["blue", "blue"], ["red", "red"]], "bankroll": [1077, 1005, 1144, 831, 841, 1102], "hand": 1, "pnl": [82, 0, 149, -164, -164, 97], "pooled...
  - seat 2 (gpt-6.2-astra): `{"belief": {"p_a": 0.231}, "limit": 25, "side": "buy", "size": 6}`
  - seat 3 (gpt-6.2-sol): `{"belief": {"p_a": 0.084}, "limit": 20, "side": "sell", "size": 5}`
  - seat 4 (gemini-4-argon): `{"belief": {"p_a": 0.21}, "limit": 17, "side": "buy", "size": 1}`
  - seat 5 (gemini-4-carbon): `{"belief": {"p_a": 0.195}, "note": "6 @ 21 rules out >5-size 3B buyers (little b>=10) and rules out heavy <=16 supply (little b<=7) => b~8 (12.3%) and b=9 (27.5%), p_a~0.195. Kelly at c~20-21 is ~0; pass R2.", "side":...`
- **exchange** largest_belief_logloss: claude-sonnet-5.5 seat 2 = 0.0106 (`exchange/full-01/s1/g0/r3/gemini-4-carbon+claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra+gpt-6.2-sol+gemini-4-argon`)
- **exchange** largest_belief_logloss: claude-sonnet-5.5 seat 4 = 0.003454 (`exchange/full-03/s1/g0/r1/gpt-6.2-sol+gemini-4-argon+gemini-4-carbon+claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra`)
- **exchange** largest_belief_logloss: gemini-4-argon seat 1 = 0.002709 (`exchange/full-06/s1/g0/r1/gpt-6.2-sol+gemini-4-argon+gemini-4-carbon+claude-fable-5.2+claude-sonnet-5.5+gpt-6.2-astra`)
