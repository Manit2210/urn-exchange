# Urn Exchange: rules and grading

Code: `src/urn_exchange/exchange.py` (game id `exchange`, version 0.1.0, 3-6 players), exact maths shared with
the solo game in `src/urn_exchange/urns.py`. Tests: `tests/test_exchange.py`. Power analysis: `scripts/power.py`.
Paper chips, research only, not investment advice.

Urn Exchange tests Bayesian updating, the gap between a belief and a price, sizing, passing and paying for
information, in a market with no house: the price is set by the other players, every contract bought is sold
by another player, and the market's price is itself evidence about the other players' private balls. It is
decided by chips, like poker, and graded against exact posteriors, which poker cannot do.

## Match

10 independent hands; every seat starts with 1,000 chips. Zero-sum: all chips move between players (fills and
research fees), so payoffs sum to exactly 0 (tested on every deal at 3, 4, 5 and 6 players).

Each hand (same urn maths as `urns`: prior 0.2-0.8 from `urns.PRIORS`, red shares from `urns.COMPOSITIONS`):

1. The dealer picks urn A with the stated prior, else B. Contract pays 100 if A, 0 if B.
2. Everyone sees 1 public ball. Each seat sees its own private balls (1 or 2 per hand, the same count for every
   ordinary seat, stated), different balls from everyone else's. Balls are drawn with replacement.
3. Research phase (research families only, simultaneous): each seat may pay the hand's fee for one more private
   ball. Afterwards everyone learns how many seats bought research (anonymous; the fee shares a seat receives
   reveal the count anyway).
4. Round 1 (simultaneous sealed orders): `{"side": "buy"|"sell"|"pass", "size": 0..10, "limit": 1..99,
   "belief": {"p_a": 0..1}, "note": <=200 chars}`. `limit` may be omitted only on a pass. One call auction.
5. Round 2: everyone sees the round-1 price and volume (or "no trade" with the best bid and ask) and its own
   fill, states a new belief and sends a second order; a second auction at its own single price.
6. Settle at 100 if A else 0. Each seat then sees the urn, its own fills and P&L, both prices and its fee
   flows. Other seats' balls and fills are never shown (not even after the hand); `result().info` reveals
   everything after the match.

Decisions per seat per match: 20 (no research) or 30 (research families). Illegal actions do not count; after 3
in a row the loop applies the default (no research / pass, belief = the prior).

## Design choices

- **Duplicate play.** Every ball is a pure function of (deal, seed, hand, seat SLOT, ball index), the urn of
  (deal, seed, hand), the hand parameters of (deal, hand). The ball streams do not depend on the table size, so
  slot 0's balls are the same at 3, 4, 5 or 6 players. The schedule rotates the models through the slots
  (`schedule.py`), so every model plays every slot's evidence once, including the insider slot.
- **Call auction price.** The price p in 1..99 maximises matched volume min(D(p), S(p)) with D(p) = buy size with
  limit >= p and S(p) = sell size with limit <= p. Ties: smallest imbalance |D - S|, then the price nearest the
  middle of the tied range (the tied set is an interval, tested), an exact half-way tie going to the price nearer
  50. The rule reads only prices and sizes, so it is seat-symmetric, and it is mirror-symmetric (p -> 100 - p with
  buy <-> sell gives 100 - p, tested). No crossing: "no trade", best bid and best ask announced.
- **Rationing.** The short side fills in full. The long side fills by price priority (more aggressive limits
  first); at the marginal limit the remainder is shared pro rata to size, rounded down; the leftover contracts go
  one each by largest fractional remainder, then by a deal-stream permutation of the slots keyed (deal, seed,
  hand, round), never by seat index alone. One order per seat per round, so no self-trade is possible.
- **Research fee split.** The fee F is split among the n - 1 other seats: each gets floor(F / (n - 1)); the
  remainder r goes one chip each to the first r seats of a deal-stream permutation keyed (deal, seed, hand,
  payer slot). Whole chips, exactly zero-sum, deterministic, symmetric across rotations (tested). A seat may buy
  research only if its bankroll stays >= 1 after the fee.
- **Bankroll rule.** An order is legal if, filled in full at its limit (the worst fill price), the worst case of
  the seat's whole position this hand (round-1 fills included) leaves at least 1 chip. Bankrolls never go below 1.
- **Information.** Fills are anonymous; the insider's identity is public (stated in the rules) so the other seats
  can weigh the price accordingly.

## Deal families (each stated honestly in that deal's rules)

| family | what changes |
|---|---|
| clean | 1 public + 1 or 2 private balls per seat, exact reports, no research |
| research | as clean plus a research phase: one more private ball for a fee per hand from {3, 8, 15, 25} |
| insider | one insider slot (public, fixed per deal and table size) sees 3 private balls each hand; everyone else 1 |
| noisy | every reported colour (public, private, research) flipped with a stated probability (10-20% per deal) |
| insider_research, noisy_research | combinations |

Deals: `pilot-01..04` = clean, research, insider, noisy (15%). `full-01..12` cycle the six families twice; noise
from a deal stream.

## Grading (exact only given what each measure conditions on)

In a multiplayer game there is no exact optimal strategy: the best order depends on the other players'
strategies. So the **ranking is by chips** (duplicate BT, `ratings.multi_units`: chips summed over the rotations of
a group, each pair of models compared win/draw/loss). The graded measures below are exact only given the seat's
own information or the realised price; they explain the chips, they do not replace them.

- **Beliefs** (`belief_logloss` = mean KL(p* || stated) in nats, stated clipped to [0.001, 0.999]; `belief_brier`):
  research-phase and round-1 beliefs against the exact posterior of the seat's own information (prior, public
  ball, own private balls, own research ball once bought; noise rate). `urns.posterior` is reused; tested against
  brute-force enumeration of every colour and misreport and by Monte Carlo calibration.
- **Market read** (round 2): `market_read` = KL(pooled || stated round-2 belief), the pooled posterior using every
  seat's balls (public, private, research bought). `market_gain` = KL(pooled || own posterior) - KL(pooled ||
  stated): > 0 when the round-2 belief moved from the own-information posterior toward the pooled one (what the
  price could teach), < 0 when it moved away. `r2_belief_vs_own` keeps the KL to the own posterior.
- **Trade regret** (milli-nats of log bankroll, summed): per round with a clearing price c, the best Kelly trade at c
  (either side, any legal size 0..10 at c, given the round-1 position in round 2) minus the value of the fills the
  seat actually received (0 regret when no price formed). Round 1 is valued on the own-information posterior
  (exact: nothing else is known when the order is sent). **Round 2 is valued on the pooled posterior**, because
  the round-1 price carries the other seats' information: grading round 2 on the own posterior would penalise a
  seat that correctly learned from the price and reward one that ignored it. The pooled posterior is the
  "fully revealing price" benchmark; nobody can reach it exactly, but duplicate rotation gives every model the
  same slots. `trade_regret_r1`, `trade_regret_r2` and `trade_regret_r2_own` (round 2 on the own posterior) are
  reported separately. Rationing can add regret through no fault of the seat; it is symmetric over rotations.
- Counts on rounds with a price: `wrong_direction` (bought above / sold below the round's yardstick posterior),
  `pass_when_should_trade` (no fill while the best trade at c beat no trade by > 0.002 nats),
  `trade_when_should_pass` (filled, and worse than no trade by > 0.002 nats).
- Also `draws` (research bought), `fees_paid`, `fees_received`, `ev_chips` (fills valued at the end-of-hand pooled
  posterior, net of fees: chips with most of the urn luck removed), `final_chips`, `volume`, `trades`,
  `decisions`, `hands`, `insider`.

`HEADLINE` = chips (higher), belief_logloss (lower), market_read (lower), trade_regret (lower); `primary` = chips.

## Bots

`random` (random research, side, size, limit and belief), `alwayspass`, `bayes` (own-information posterior; a
one-sided Kelly order against the prior mid in both rounds: limit at its belief, size = Kelly at the halfway
price; ignores the market price; buys research when `urns` would pay that fee against a fair quote at the prior
mid, a heuristic), `halfkelly` and `noisy` (bayes at half size; bayes with log-odds belief noise SD 0.6),
`follower` (bayes in round 1; in round 2 states the round-1 price as its belief and passes), `pooled` (privileged
oracle using every seat's balls, Kelly against the round-1 price; the strong reference).

Duplicate rotations over the 16 deals x 3 seeds (table pooled, bayes, follower, random, alwayspass): pooled
gains, random loses, and bayes beats random in more than 75% of deals (tested). Every belief/market measure
orders as expected (bayes belief_logloss 0, pooled market_read ~0, follower market_gain > 0 > random).
A lesson the game teaches: at a table with an informed player, alwayspass (~+90 chips per deal, all from fees
received) beats bayes (~-170): trading on your own balls against a better-informed counterparty loses.

## Power analysis: how many deals separate two players?

`scripts/power.py` (offline, bots only, a few minutes on 8 cores). One deal = one (deal, seed) block
played by a 5-seat table in all 5 rotations (50 hands per seat). Pool: 12 full deals x seeds 0-39 = 480 blocks
per table; 200 random draws of N blocks per grid point; 300 bootstrap resamples over blocks. Separation tests:
paired bootstrap 95% CI of the per-deal chip difference excludes 0; BT (the ratings' fit and prior) 95% CI of the
Elo difference excludes 0 ("BT diff"); the two players' own 95% Elo intervals do not overlap ("BT disjoint", what a
reader of the report sees). A `null` table seats two copies of bayes to measure false positives.

N80 = smallest number of deals with >= 80% power (grid 4..128):

| table | pair (A > B) | mean chips A-B per deal | SD | A wins deal | N80 chips bootstrap | N80 BT diff | N80 BT disjoint | normal approx |
|---|---|---|---|---|---|---|---|---|
| spread | pooled > bayes | 2053 | 1834 | 88% | 6 | 12 | 24 | 7 |
| spread | bayes > random | 1328 | 1423 | 85% | 12 | 12 | 24 | 9 |
| spread | bayes > follower | 143 | 556 | 66% | 128 | 96 | > 128 | 118 |
| spread | bayes > alwayspass | -256 | 731 | 36% | 96 | 96 | > 128 | 64 |
| close | bayes > random | 1448 | 1762 | 82% | 16 | 16 | 24 | 12 |
| close | bayes > noisy06 | 461 | 929 | 71% | 32 | 48 | 64 | 32 |
| close | noisy03 > noisy06 | 380 | 696 | 72% | 32 | 48 | 64 | 27 |
| close | bayes > noisy03 | 81 | 759 | 53% | > 128 | > 128 | > 128 | 688 |
| close | bayes > halfkelly | -43 | 536 | 48% | > 128 | > 128 | > 128 | 1245 |
| null | bayes > bayes2 | -26 | 573 | 47% | > 128 | > 128 | > 128 | 3891 |

Tables: spread = pooled, bayes, follower, random, alwayspass; close = bayes, noisy03, noisy06, halfkelly, random;
null = bayes, bayes2, noisy03, halfkelly, random. False-positive rate (null pair) by N: chips bootstrap 26% at 4
deals, 12% at 6-12, 8-9% at 16-32, ~5% from 48; BT diff 20% at 4, 7-9% at 6-32, ~5% from 48; BT disjoint <= 1%
from 8. So the chips bootstrap is anti-conservative with few deals; read BT diff, and treat < 8 deals as noise.

Reading: a large skill gap (strong vs weak, ~1,300-2,000 chips per deal) separates in about **12 deals** (BT diff;
24 for disjoint report intervals). A close gap like exact beliefs vs belief noise of SD 0.6 in log-odds (~400-460
chips per deal, the deal won ~71% of the time) needs about **48 deals** (64 for disjoint intervals). Smaller gaps
(noise SD 0.3, half vs full Kelly) are not separable in any practical run: their true chip difference is near 0
(half Kelly even edges full Kelly at these tables). Caveat: these are fixed 5-seat tables of bots. v1 seats
six models at one six-seat table, so every pair of models meets in every deal; its 12 deals separate large gaps
only, and about 48 deals (12 full deals x seeds 0-3) would be needed for close models.

## Size and cost

Rules are about 3.0k characters (limit 3,500), views 300-550 characters (limit 700), both tested. Each seat makes
20 decisions per match (30 in research families), one tool call each, and the conversation grows over the match.
v1 (six models, 72 matches) made about 1,800 calls per model and cost USD 371 for the 72 final matches; see the
README for cost and latency per model. `urn-exchange dry-run` measures call counts and message sizes with bots
for free and prints an estimate.

## Open points

- Bots are thin markets: two bayes seats only trade when their posteriors fall on opposite sides of the prior.
  Models may quote more; nothing in the rules asks them to.
- Research VOI in a market has no exact value (it depends on what the price will reveal), so research is graded
  only through chips and the fee counts.
