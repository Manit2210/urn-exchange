# Urn Exchange v1 (18 Nov 2026): results and trace review

Run `games-exchange-20261118T131944-f9c72eed` (dataset `v1` in the trace file): Urn Exchange 0.1.0, six models
at one 6-seat table, 12 deals (full-01..06 x seeds 0-1), each deal played in all 6 seat rotations, 72 matches, all
finished, no forced passes. GPT-6.2 Astra and Sol and Claude Fable 5.2 and Sonnet 5.5 at reasoning effort `high`
(through OpenRouter, on the publisher's own provider keys via BYOK); Gemini 4 Argon and Carbon with thinking capped
at 8k (partly direct on the Gemini API, then through OpenRouter, same cap). Reply limit 16k. Cost USD 371 for the
72 final matches, USD 416 including matches re-played after provider errors. Config: `configs/v1.yaml`. Paper
chips, research and fun only, not investment advice.

## Results

| model | chips | 95% range (bootstrap over deals) | luck-free ev_chips per match |
|---|---|---|---|
| Gemini 4 Carbon | +4,612 | +548 to +8,839 | +112 |
| Gemini 4 Argon | +3,619 | +788 to +6,528 | +12 |
| Claude Sonnet 5.5 | -315 | -3,806 to +3,270 | -8 |
| Claude Fable 5.2 | -1,635 | -4,826 to +1,541 | -48 |
| GPT-6.2 Sol | -2,675 | -6,235 to +916 | -55 |
| GPT-6.2 Astra | -3,606 | -8,411 to +1,191 | -12 |

`ev_chips` values each fill at the pooled posterior (every player's balls), so it removes the luck of the draw.
Only the two Gemini models are clearly positive on chips; the order of the other four is not settled. On luck-free
value the order is Carbon, Argon, Sonnet, Astra, Fable, Sol.

## What the traces show (three independent reviews of the traces, one per lab's models)

**Everyone computes beliefs correctly.** Belief error against each player's own exact posterior is about zero for
all six. The game was decided by when and how much to trade against better-informed players, not by the maths.

**Gemini 4 Carbon: the real winner, by abstaining.** It passes 29% of round-1 orders (Astra 3%, Sol 7%, Fable 9%,
Sonnet 18%, Argon 10%) and 81% when its own private balls cancel. Its notes: "zero private edge, pure adverse
selection either side. Pass R1." It rarely bets max size (2% of R1 orders vs 11-13% for Astra and Fable). Nearly all
its profit is round 1 (+5,858); its moderate-belief round-2 trades lose (-1,280), and it sometimes over-decodes a
single fill (worst hand: "E[p_all]~0.974" from one fill, then bought more).

**Gemini 4 Argon: good, but mostly lucky here.** ev +12 per match against +50 realised: about three quarters of its
+3,619 is variance (the same reasoning on one noisy hand won +1,030 in one seat and lost -810 in another). It is
positive in every deal family and reads the round-1 price well in round 2 (+3.2 per contract). An early impression that
Argon habitually sits out round 1 was wrong: that was one note.

**Reading the price is common, not decisive.** Round-2 beliefs move toward the pooled posterior equally for the two
Gemini and two GPT models (82-85%); only the Claude models read it worse (Fable 76%, Sonnet 64%).

**Claude Sonnet 5.5: cheap caution.** It defers to the market: 70% of its round-2 beliefs sit within 3 points of the
round-1 price and it passes 68% of round-2 decisions (Fable 29%). That cost little overall (ev -596 total vs Fable
-3,456) but loses on trap hands where the price itself is wrong. Its insider-deal win (+1,526) and noisy-deal loss
(-1,695) are mostly variance (ev +190 and -764); noisy maths was correct. An early "gambler's fallacy" reading was not
borne out: earlier hands never moved Sonnet's beliefs.

**Claude Fable 5.2: over-trades round 1, then fails to reverse.** Research-family round 1: ev -2,290 on 463
contracts. In 34 hands it filled in round 1, recognised the mistake in round 2 and tried to flatten, but did not fill:
-1,958 chips. Research fees were not the problem (net +93). It cites earlier hands as evidence more than anyone (93
calls), weakly associated with losses. Its noisy-deal win is one hand.

**GPT-6.2 Sol: the structural loser.** ev -55 per match; it was about 1,270 chips lucky. Round-1 trading loses 3.3
chips per contract against what the table knew, -8.9 per contract against the insider. Its notes name adverse
selection ("a fill against the red bidders would be informative, so I limit size") but it trades anyway.

**GPT-6.2 Astra: never passes, winner's curse, and research fees.** Realised -3,606 but ev only -12 per match, almost
all research fees (net -813); its trades are roughly break-even, so much of the realised loss is variance. By its own
beliefs its round-1 fills held about 26,000 chips of edge; at the pooled posterior they held +880. Its "orders against
its own belief" were harmless round-2 unwinds. It wrote 0 notes in 1,440 orders; reasoning is encrypted, so why it
never passes is not visible.

**Who funds whom.** Both Gemini models take most from the GPT models, especially Sol; Carbon also takes from Argon.

## Caveats

12 deals separate large gaps only (power analysis in `rules.md`); one noisy trap hand (full-04 seed 0 hand 6)
moves several results. ev uses hindsight of all players' balls: it measures adverse selection, not an attainable
strategy. Thinking text: Claude on every call; Gemini only on the OpenRouter route (~60% of calls); GPT encrypted, with short summaries on about a third of calls.
A one-deal side run with Claude and GPT at `xhigh` (Claude thought 3-5x more, GPT about 2x) moved individual results (GPT-6.2 Sol most) and a Gemini model still finished first; one deal is too few to change the ranking.
