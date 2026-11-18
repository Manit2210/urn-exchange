# Harness: what a model sees, calls and spend, schedule, ratings, data

Code: `src/urn_exchange/{runner,schedule,ratings,summary,traces,cli}.py` and `src/urn_exchange/providers/`.
Paper chips, research only, not investment advice.

## 1. What a model sees (GAMES_WIRE 1)

One conversation per seat per match. Every match is driven by `core.play_match`, so models and bots get
the same views and the same validation (`Game.check`).

- **System prompt** = `ARENA_PREAMBLE` + `game.rules(seat)` (`runner.system_prompt`). The preamble says:
  paper chips, for fun; you are seat X of N in the game; the other seats are other models or bots and you
  see only what your seat may know; act only with the `act` tool, once per turn; put your honest belief in
  every action (graded against the exact answer where one exists); an illegal action comes back as an
  ERROR and after 3 in a row the game plays a default; you are scored on chips and on the quality of
  decisions and beliefs. The preamble names the arena "DeskBench Games" (the name of the game suite Urn
  Exchange was built in) and numbers the seat from 0, while the rules number seats S1-S6. It is kept
  byte-identical to what the v1 models saw; change it only together with `GAMES_WIRE`.
- **Tools**: exactly one, `act`, whose parameters are `game.action_schema(seat)` as authored (research
  phase: `research` true/false and `belief`; rounds: `side`, `size`, `limit`, `belief`, optional `note`).
- **Messages**: the first view is a user message. After the model calls `act`, the next view is the tool
  result answering that call (canonical `role: tool`; every adapter converts it: OpenRouter/Chat
  Completions `tool` messages, Responses `function_call_output`, Gemini function responses, Anthropic
  `tool_result`). Extra calls in the same turn get "Ignored: only one act call per turn..."; the first
  `act` is applied. A wrongly named tool gets an "Unknown tool" result that carries the nudge.
- **Simultaneous moves**: in each phase every seat is asked in turn, but each sees only its own information
  and no order is revealed until all seats have sent theirs; then the call auction clears.
- **No tool call**: one nudge (a user message), then `act` returns `{}`; the game rejects it and the loop
  applies the default after 3 in a row. Messages strictly alternate.
- **Reasoning state**: the assistant turn is appended exactly as returned, so `provider_state` (OpenRouter
  `reasoning_details`, Gemini thought signatures, Anthropic thinking blocks) is echoed back unmodified, as
  the providers require for multi-turn tool use.
- **Protocol**: the same `max_tokens` (16,000) and the same reasoning effort (`high`) for every model: the
  config's `params` are applied after each model's own. A per-model thinking cap (`reasoning: {max_tokens:
  8000}`, used for both Gemini models in v1) replaces the effort for that model, because OpenRouter accepts one
  of the two. No python tool: the game tests reasoning in the model's own head.

## 2. Calls, retries, spend, cache

- **Retries**: rate limits (429, including Google AI Studio's spend-based limit, which gets a 90 s wait) back
  off and retry up to 8 times; transient errors (5xx, gateway errors, dropped connections, timeouts) up to 4
  times. Auth, missing keys, exhausted quota and insufficient credits halt the run at once.
- **Pacing**: `rpm` per model (client side, per process).
- **Spend guard**: every paid call reserves its worst case (input estimated from the message size, the full
  `max_tokens` of output, no cache discount) on a SQLite ledger (`ledger`, default `results/spend.sqlite`)
  and refuses the call if the run's spend plus that reservation would pass `budget_usd`. After the call the
  reservation becomes the billed cost; rejected requests are released. The budget is scoped to the run id,
  so resumes keep counting.
- **Billed cost**: on OpenRouter the runner books OpenRouter's reported `usage.cost`; under BYOK (your own
  provider key added in OpenRouter) it books the upstream inference cost plus OpenRouter's fee. Direct
  providers are booked at the price table (`pricing.py`).
- **Cache and resume**: every response is stored under `cache_dir`, keyed by the provider request hash over
  the messages, tool, system prompt and params plus the game, version, deal, seed, seat, match and run id.
  So different runs, seats and seeds never replay each other; only a resumed match of the same run replays
  its own calls. `urn-exchange run --resume RUN_ID` reuses the run directory, skips matches already `ok`
  under the same game version and wire, and re-plays the rest (cached calls are free).
- **Circuit breaker**: a model with `circuit_breaker` errored matches in a row is skipped for the rest of
  the run (those matches are recorded as `skipped`; resume later).

## 3. Schedule

For each (deal, seed) the roster is split into tables of `n_players` models (one table in v1: six models,
six seats). Each table plays `n_players` matches on that deal with the models rotated cyclically through the
seats, so every model plays every seat's evidence once (duplicate play), including the insider seat. The
rotations share a `unit_id` (the duplicate group). With more models than seats, tables are chosen greedily so
every model plays equally often and every pair meets about equally often. With fewer models than seats, the
table is padded with `bot:<fill_bot>` seats. Deals: an explicit list, or `split` (pilot/full) + `n_deals`.
`rotations` limits a smoke test to the first rotations. Deterministic; match ids are unique.

## 4. Ratings

Reads only match records (a run directory or a trace file), never the ledger.

- **Chips** (the headline ranking): each model's chips summed over every match, with a 95% percentile
  bootstrap range over duplicate deals (the 12 per-deal totals resampled with replacement, 4,000 times,
  fixed seed).
- **Luck-free value** (`ev_chips`): every fill valued at the pooled posterior of all players' balls at the
  end of the hand (instead of at 100 or 0), net of research fees. It removes the luck of the urn draw but uses
  hindsight of every player's balls: it measures adverse selection, not an attainable strategy.
- **Elo (Bradley-Terry)**: per complete duplicate group, each model's chips summed over the rotations; every
  pair of models in the group is one comparison (more chips = win, equal = draw; margins do not enter).
  Penalised maximum likelihood (Newton), draws count half, Gaussian prior SD 600 Elo so a perfect record
  stays finite; Elo = 1500 + 400/ln 10 x (t - mean t) (1500 = roster mean). 95% intervals: percentile
  bootstrap over groups, 1,000 resamples, fixed seed. Elo counts deals won, chips count how much: in v1 Argon
  wins slightly more head-to-head deals, Carbon wins more chips.
- **Skill profile**: per model, the mean of each headline metric (chips per match, belief_logloss,
  market_read, trade_regret) with a bootstrap range over deals, and its rank.
- **Highlights**: the matches with the largest chip swings (last actions and revealed deal) and the largest
  belief errors.
- Outputs (`urn-exchange ratings`): `report.md`, `ratings.json` (strict JSON), `ratings.csv`, `chips.csv`,
  `profile.csv`, `matches.csv` (one row per match and seat) and `summary.json` (per model: chips and range,
  ev, deals won, by family, behaviour, calls, tokens, latency, cost, Elo).

## 5. Data and trace format

A run directory (`results/runs/<run_id>/`) holds `manifest.json` (config, game metadata, schedule, run
summary), `events.jsonl` (run and match events) and `matches.jsonl`: one JSON record per finished match
attempt (a resumed run appends; the last record per match id wins). A trace file
(`urn-exchange traces`, `.jsonl.gz`) holds the last `ok` record per match with opaque reasoning blobs removed
and a `dataset` label. Each record:

| field | content |
|---|---|
| `dataset` | trace files only: `v1` (main run) or `v1-xhigh` (side run) |
| `match_id`, `unit_id`, `rotation` | `exchange/<deal>/s<seed>/g<group>/r<rotation>/<seat 1 model>+...+<seat 6 model>`; the duplicate group; the seat rotation |
| `deal_id`, `seed`, `game_id`, `game_version`, `games_wire` | what was played, under which rules and message wire |
| `seating` | model key per seat (seat 1 first) |
| `status`, `started_at`, `finished_at`, `wall_s` | outcome and timing |
| `steps` | every decision in order: `seat` (0-based), `view` (the exact text the seat received; null on a retry), `action` (what it sent, including `belief.p_a` and any `note`), `error` (null if legal), `defaulted` |
| `invalid`, `defaulted` | illegal actions and forced defaults per seat (v1: 37 illegal actions, all fixed on the retry; no forced defaults) |
| `result.payoffs` | chips won or lost per seat (zero-sum) |
| `result.metrics` | per seat: `chips`, `ev_chips`, `final_chips`, `belief_logloss`, `belief_brier`, `market_read`, `market_gain`, `r2_belief_vs_own`, `trade_regret` (+ `_r1`, `_r2`, `_r2_own`), `wrong_direction`, `pass_when_should_trade`, `trade_when_should_pass`, `draws`, `fees_paid`, `fees_received`, `trades`, `volume`, `decisions`, `hands`, `insider` |
| `result.info` | the revealed deal: family, noise, insider seat, and per hand the urn, the spec, every seat's balls, research bought, the pooled posterior, both auction rounds (price, volume, fills, best bid/ask), P&L and bankrolls |
| `result.transcript` | the public log (research counts, clearing prices and fills, settlements) |
| `players[]` | per seat: `model_key`, `provider`, `model`, `system` (the full system prompt), `summary` (calls, tokens, cost, nudges, truncations, cache hits) and `calls[]` |
| `players[].calls[]` | per model call: `kind` (act/nudge), `turn`, `text` (visible reply), `tool_calls`, `usage` (input, cached input, output tokens incl. reasoning), `cost_usd` (true cost), `billed_cost_usd` (OpenRouter bill; the BYOK fee only under BYOK), `latency_s`, `wall_s`, `stop_reason`, `truncated`, `upstream_provider`, `cache_hit`, `request_hash`, `started_at`, `provider_state` |
| `provider_state` | readable reasoning kept: OpenRouter `reasoning_details` of type `reasoning.text` (Claude, Gemini) and `reasoning.summary` (GPT); removed: `reasoning.encrypted` data (`data_removed: true`), thinking-block `signature` (`signature_removed`), Gemini `thought_signature` (`thought_signature_removed`) |

Readable thinking in v1: Claude on every call (full thinking text); Gemini only on calls routed through
OpenRouter (about 60% of its calls; the direct Gemini API route returns signatures only); GPT as short
reasoning summaries on about a third of calls (the full reasoning is encrypted).

## 6. Commands

```
urn-exchange dry-run  --config configs/v1.yaml
urn-exchange schedule --config configs/v1.yaml --matches
urn-exchange run      --config configs/smoke.yaml [--yes] [--resume RUN_ID]      # paid
urn-exchange ratings  --run results/runs/<run_id>
urn-exchange ratings  --run traces/urn-exchange-v1-traces.jsonl.gz --dataset v1 --out /tmp/v1-report
urn-exchange traces   --run results/runs/<run_id> --out traces/<name>.jsonl.gz --dataset <label>
urn-exchange play     --deal pilot-02 --bots bayes,follower,random --seat 1
pytest -q
```
