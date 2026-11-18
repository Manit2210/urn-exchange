# Urn Exchange

A zero-sum trading game for language models, graded against exact Bayesian posteriors.

Six models sit at one table and trade a contract on a hidden urn. Each sees a few private draws, so every
trade is against someone who may know more. Chips decide the ranking, as in poker, but because the urn
maths is exact, every belief and every trade can also be graded with the luck taken out.

Write-up: https://continualawareness.com/research/urn-exchange

Paper chips only. This is research, not investment advice.

## The game

Each match is 10 independent hands. Every player starts with 1,000 chips.

1. The dealer secretly picks urn A (with a stated prior, for example 50%) or urn B. Each urn has a stated
   share of red balls, for example A 70% red and B 30% red.
2. Everyone sees one public ball. Each player also sees its own private balls (1 or 2, or 3 for the insider
   in insider deals), which nobody else sees.
3. Some deals have a research phase: pay a fee (3 to 25 chips, shared out to the other players) for one more
   private ball.
4. Round 1: every player sends one sealed order at the same time: buy, sell or pass, a size (1 to 10
   contracts) and a limit price (1 to 99). A call auction finds the single price that matches the most
   contracts. Everyone learns that price and volume.
5. Round 2: a second sealed order and a second auction. The round-1 price is evidence about the other
   players' balls.
6. The contract pays 100 if the urn is A and 0 if B. There is no house: every contract bought was sold by
   another player, so chips only move between players.

Every order also carries the player's probability that the urn is A (`belief.p_a`) and an optional short
note. Some deals add noise (each reported colour may be flipped) or an insider seat with three private balls.
The full rules are in [docs/rules.md](docs/rules.md); the exact text each model saw is in the traces.

### A worked hand

Three players. Prior 50%, urn A 70% red, urn B 30% red, one private ball each. The public ball is red.

| | Private ball | Exact P(A) from its own balls | Round-1 order |
|---|---|---|---|
| S1 | red | 84.5% (two reds) | buy 4, limit 80 |
| S2 | blue | 50% (one red, one blue) | sell 3, limit 55 |
| S3 | blue | 50% | sell 2, limit 60 |

The auction: the buyer wants 4 at any price up to 80; sellers offer 3 from 55 and 5 from 60. Every price from
60 to 80 matches 4 contracts, so the price is the middle, **70**. S1 buys 4; S2 sells 3 (better limit first)
and S3 sells 1.

Pooling all four balls (two red, two blue) gives P(A) = 50%. S1 paid 70 for a contract worth 50 on everything
the table knew: the sellers' eagerness was a sign that they held blue balls. A good player reads that from
the price in round 2; here everyone passes.

The urn turns out to be A. S1 wins 4 x (100 - 70) = **+120**, S2 loses 90 and S3 loses 30. But valued at the
pooled posterior, S1's trade was worth 4 x (50 - 70) = **-80**. Chips say S1 played well; the luck-free value
(`ev_chips`) says S1 was adversely selected and got lucky. Over many hands the two converge, and the grading
tells them apart long before they do.

## Why it is interesting

- **Skill is separated from luck exactly.** The posterior of any set of balls is computable, so every belief is
  graded against the exact answer for the player's own information, and every fill is valued at the pooled
  posterior of all players' balls.
- **Models play each other, not a fixed answer key.** The price is set by the other players, so the test gets
  harder as the field gets stronger.
- **Duplicate play.** Every deal is played once in each seat rotation, so every model gets every seat's balls,
  including the insider seat. Luck from who sat where cancels.
- **It tests trading judgement, not arithmetic.** Adverse selection (who is on the other side of my fill?),
  sizing, when to pass, whether information is worth its fee, and how much to learn from a price.

## v1 results (18 November 2026)

Six models, one six-seat table, 12 duplicate deals (6 deal families x 2 seeds), each played in all 6 seat
rotations: 72 matches, 720 hands per model, all matches finished, no forced passes. Reasoning effort `high`
for GPT-6.2 and Claude; thinking capped at 8,000 tokens for Gemini. Reply limit 16,000 tokens.

| Rank | Model | Setting | Chips | 95% range | Luck-free value per match | Elo (95% range) |
|---|---|---|---|---|---|---|
| 1 | Gemini 4 Carbon | thinking 8k | **+4,612** | +548 to +8,839 | +111.8 | 1605 (1490 to 1746) |
| 2 | Gemini 4 Argon | thinking 8k | **+3,619** | +788 to +6,528 | +11.6 | 1616 (1519 to 1769) |
| 3 | Claude Sonnet 5.5 | effort high | -315 | -3,806 to +3,270 | -8.3 | 1520 (1413 to 1643) |
| 4 | Claude Fable 5.2 | effort high | -1,635 | -4,826 to +1,541 | -48.0 | 1459 (1346 to 1561) |
| 5 | GPT-6.2 Sol | effort high | -2,675 | -6,235 to +916 | -54.8 | 1406 (1279 to 1510) |
| 6 | GPT-6.2 Astra | effort high | -3,606 | -8,411 to +1,191 | -12.4 | 1395 (1200 to 1541) |

Chips are summed over all 72 matches (zero-sum: the column adds up to 0); the range is a bootstrap over the
12 duplicate deals. Elo counts head-to-head deals won (who finished a deal ahead of whom) and chips count by
how much, so Argon won slightly more pairings while Carbon won more chips.

What we found (details in [docs/findings-v1.md](docs/findings-v1.md)):

- **Only the two Gemini models are clearly positive.** The order of the other four is not settled by 12 deals.
- **Everyone does the maths.** Belief error against each player's own exact posterior is about zero for all
  six. The game was decided by when and how much to trade against better-informed players.
- **Gemini 4 Carbon wins by abstaining.** It passes 29% of round-1 orders (GPT-6.2 Astra 3%), and 81% when
  its own private balls cancel out. In its own note: "zero private edge, pure adverse selection either side".
- **Gemini 4 Argon is good but mostly lucky here:** luck-free value +12 per match against +50 realised.
- **Claude Sonnet 5.5 defers to the market:** it passes 68% of round-2 decisions, which costs it little.
- **Claude Fable 5.2 over-trades in round 1** and often fails to unwind in round 2.
- **GPT-6.2 Sol is the structural loser** (luck-free value -55 per match): its notes name adverse selection,
  but it trades anyway. **GPT-6.2 Astra almost never passes** and pays the most research fees; its trades
  are roughly break-even, so much of its loss is variance.
- A one-deal side run with GPT-6.2 and Claude at effort `xhigh` (Claude thought 3-5x more, GPT about 2x) moved
  individual results, but a Gemini model still finished first.

### Cost and latency per model (v1)

| Model | Calls | Output tokens per move (median) | Latency per move, median / p90 | Cost (USD) | Per match |
|---|---|---|---|---|---|
| Gemini 4 Carbon | 1,848 | 4,802 | 74 s / 149 s | 199.84 | 2.78 |
| Gemini 4 Argon | 1,811 | 753 | 23 s / 56 s | 37.91 | 0.53 |
| Claude Sonnet 5.5 | 1,801 | 201 | 3.1 s / 4.7 s | 13.48 | 0.19 |
| Claude Fable 5.2 | 1,804 | 378 | 4.7 s / 9.3 s | 75.46 | 1.05 |
| GPT-6.2 Sol | 1,802 | 342 | 8.0 s / 14.5 s | 14.10 | 0.20 |
| GPT-6.2 Astra | 1,800 | 122 | 6.3 s / 11.8 s | 30.21 | 0.42 |
| **Total** | | | median match: 54 min | **371.00** | |

Output tokens include reasoning. Costs are what was billed for the 72 final matches; matches re-played after
provider errors brought the whole run to about USD 416.

## Reproduce it

You need Python 3.11 or later and your own API keys. Nothing here calls a model unless you run
`urn-exchange run`.

```bash
git clone <this repository> urn-exchange && cd urn-exchange
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # add [gemini] for the direct Gemini route, [anthropic] for direct Claude
pytest -q                   # offline, no keys, about 10 seconds
```

**Keys** are read from environment variables only (see [.env.example](.env.example)); never put a key in a
config file. v1 runs every model through [OpenRouter](https://openrouter.ai), so one key is enough:

```bash
export OPENROUTER_API_KEY=...   # required for configs/v1.yaml and configs/smoke.yaml
export GEMINI_API_KEY=...       # only for the optional direct-Gemini route
```

If you would rather pay the model providers directly, add your OpenAI, Anthropic or Google keys in your
OpenRouter settings (BYOK, bring your own key). OpenRouter then routes each call to that provider on your key
and charges only its BYOK fee; the runner still books the true cost (provider cost plus fee). v1 ran GPT-6.2
and Claude this way. The configs pin each model to its own lab's host with no fallbacks.

**1. Dry run (free).** Scripted bots play every model seat through the real message pipeline, then the
command prints a cost estimate from the measured call counts and the token profiles measured in v1:

```bash
urn-exchange dry-run --config configs/v1.yaml
```

For v1 it plays all 72 matches in a few seconds with 0 errors and estimates about USD 350 (worst case about
USD 4,000, if nothing were cached and every reply hit the 16k limit).

**2. Smoke run (about USD 0.70).** One match with three of the cheaper models on one short deal, to check your
key, routing and tool calls. Set `budget_usd` in [configs/smoke.yaml](configs/smoke.yaml) (for example 3),
then:

```bash
urn-exchange run --config configs/smoke.yaml
```

**3. Full v1 run (about USD 370-420, several hours).** Set `budget_usd` in [configs/v1.yaml](configs/v1.yaml)
(for example 450), then:

```bash
urn-exchange run --config configs/v1.yaml
urn-exchange ratings --run results/runs/<run_id>
```

A match takes about an hour (the slowest seat sets the pace) and 24 run in parallel, so expect roughly three
to five hours end to end. If anything stops the run (Ctrl-C, a provider outage, the budget), resume it:
finished matches are skipped, and calls already paid for are replayed from the local cache for free.

```bash
urn-exchange run --config configs/v1.yaml --resume <run_id>
```

**Spend safety.**

- `budget_usd` is required, and the shipped configs set it to 0, so a paid run refuses to start until you
  set it. It is a hard ceiling for the run id, resumes included: before every call the runner reserves that
  call's worst case and refuses the call if the run would go over.
- `run` asks you to type `yes` before the first paid call (or pass `--yes`).
- Also set a spend limit on your OpenRouter key and at each provider. The budget here is a guard inside this
  program, not a cap on your bill.
- **Gemini's thinking can run away.** Without a cap, Gemini at high effort can spend the whole 16k reply limit
  thinking and never act. The configs cap it at 8,000 thinking tokens (`reasoning: {max_tokens: 8000}`); keep
  that cap. Even with it, Gemini 4 Carbon hit the reply limit on 34 of its 1,848 calls in v1 (a reply without a
  move gets one nudge, then the game asks again).
- On the direct Gemini route, lower Google AI Studio tiers have a spend-based rate limit per rolling 10
  minutes (the runner waits it out) and a monthly spend cap (calls fail until you raise it).

Other options: [configs/v1_xhigh.yaml](configs/v1_xhigh.yaml) is the one-deal side run at effort `xhigh`
(about USD 46). `--models a,b` runs a slice of the roster, and `urn-exchange schedule --config ... --matches`
lists the matches without playing them.

**Play it yourself** (free, no keys) against the scripted bots, seeing exactly what a model sees:

```bash
urn-exchange play --deal pilot-02 --bots bayes,follower,random --seat 1
```

## How the ratings work

- **Chips** are the ranking. Each model's chips are summed over all matches. The 95% range comes from
  resampling the 12 duplicate deals with replacement (4,000 times).
- **Luck-free value** (`ev_chips`) values every fill at the pooled posterior of all players' balls at the end
  of the hand, net of research fees. It removes the luck of the urn draw but uses hindsight of everyone's
  balls, so it measures adverse selection, not a strategy anyone could follow.
- **Elo** is a Bradley-Terry fit on duplicate deals: in each deal every pair of models is one comparison (more
  chips over the six rotations wins). 1500 is the table average; the range is a bootstrap over deals.
- **Graded decisions**: research and round-1 beliefs against the exact posterior of the player's own balls
  (`belief_logloss`, KL divergence in nats); the round-2 belief against the posterior of all players' balls
  (`market_read`: how well the price was read); trade regret against the best Kelly trade at the realised
  price (`trade_regret`, in milli-nats of log bankroll); and counts of trading the wrong way, passing with an
  edge and trading without one.

Details: [docs/harness.md](docs/harness.md) (section 4) and [docs/rules.md](docs/rules.md) (grading).

## Data

| Path | Content |
|---|---|
| [results/v1/summary.json](results/v1/summary.json) | Per model: chips and range, luck-free value, deals won, results by deal family, behaviour, calls, tokens, latency, cost, Elo; the side run; quotes checked against the traces |
| [results/v1/matches.csv](results/v1/matches.csv) | One row per match and seat (432 rows): deal, family, seat, model, chips, `ev_chips`, every graded metric, calls, tokens, cost |
| [results/v1/matches_xhigh.csv](results/v1/matches_xhigh.csv) | The same for the `xhigh` side run (36 rows) |
| [results/v1/ratings/](results/v1/ratings/) | `report.md`, `ratings.json` and the chips, Elo and profile tables, regenerated from the traces |
| `traces/urn-exchange-v1-traces.jsonl.gz` | Every move of every v1 match (a GitHub release attachment, see below) |

**Traces.** `urn-exchange-v1-traces.jsonl.gz` (6,374,075 bytes; 31.0 MB uncompressed; SHA-256
`c26f06bd26e54c521256007dad8f6d26cf50315707ca69f489f61532dfeb1765`) is attached to the v1 GitHub release
instead of being committed; download it into `traces/`. It holds 78 match records: the 72 final matches of
the main run (`"dataset": "v1"`) and the 6 matches of the side run (`"dataset": "v1-xhigh"`). Each record has
every view a model received, every action it sent (with its belief and note), the system prompt, and per
call the reply, tool call, token usage, cost, latency and readable reasoning (Claude's thinking, Gemini's
thinking on the OpenRouter route, GPT's reasoning summaries). Encrypted reasoning and signature blobs were
removed (marked `data_removed`, `signature_removed` or `thought_signature_removed`); they are only meaningful
to the provider that issued them. The field-by-field format is in [docs/harness.md](docs/harness.md)
(section 5).

```bash
urn-exchange ratings --run traces/urn-exchange-v1-traces.jsonl.gz --dataset v1 --out /tmp/v1
```

regenerates `results/v1/ratings/` from the traces (identical up to floating-point rounding). Export your own
runs the same way with `urn-exchange traces --run results/runs/<run_id> --out traces/<name>.jsonl.gz`.

## Caveats and limitations

- **12 deals separate large gaps only.** A power analysis with bots ([scripts/power.py](scripts/power.py))
  puts a close gap at about 48 deals. Only the two Gemini models are clearly ahead, and one noisy trap hand
  (full-04 seed 0, hand 6) moves several results.
- **One roster, one table.** Results depend on who else is at the table: a model that profits from weaker
  traders could rank differently in a stronger field.
- **Settings differ by lab.** GPT-6.2 and Claude ran at effort `high`; Gemini had an 8k thinking cap instead,
  because uncapped it can think through the whole reply limit. The `xhigh` side run is a single deal.
- **Routes differ.** Some Gemini calls went direct to the Gemini API and the rest through OpenRouter, with the
  same model and cap; readable Gemini thinking exists only for the OpenRouter part (about 60% of calls).
- **Reasoning is partly hidden.** GPT reasoning is encrypted (short summaries only), so why GPT-6.2 Astra
  never passes is not visible in its traces.
- **Models drift.** Hosted models can change behind the same name; a rerun measures the models as they are
  on the day you run it.
- **The prompts are fixed.** Every model saw the same system prompt (which names the game suite "DeskBench
  Games", where Urn Exchange was built) and the same rules; other wordings may change behaviour.
- **Paper chips.** Nothing here is investment advice or evidence about real markets.

## Repository layout

```
src/urn_exchange/   exchange.py (the game and its bots), urns.py (exact maths), core.py (game contract and
                    match loop), runner.py, schedule.py, ratings.py, summary.py, traces.py, play.py, cli.py,
                    pricing.py, spend.py, cache.py, providers/ (OpenRouter and OpenAI-compatible, Gemini
                    API, OpenAI Responses, Anthropic, fake)
configs/            v1.yaml (the published run), smoke.yaml, v1_xhigh.yaml
docs/               rules.md, harness.md, findings-v1.md
results/v1/         summary, per-match tables, ratings
scripts/power.py    how many deals separate two players (bots, offline)
tests/              offline tests (no keys, no network)
```

## Citation

```bibtex
@misc{urnexchange2026,
  title        = {Urn Exchange: a zero-sum trading game for language models},
  author       = {{Continual Awareness}},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://continualawareness.com/research/urn-exchange}},
  note         = {Code, data and traces in the urn-exchange repository}
}
```

## License

Code (`src/`, `tests/`, `scripts/`, `configs/`) is licensed under the [Apache License 2.0](LICENSE). Data,
traces and documentation (`results/`, `traces/`, `docs/` and the text of this README) are licensed under
[CC BY 4.0](LICENSE-DATA). Copyright 2026 Continual Awareness.
