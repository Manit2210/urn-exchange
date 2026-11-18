"""Runner: ModelPlayer message shape, nudges, feedback, provider_state, call records, cache keys, spend
refusals, dry runs, resume, circuit breaker and halts. Offline: FakeProvider and scripted bots only."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml
from toys import BOTS, ToyDuel, toy_registry

from urn_exchange import runner as R
from urn_exchange.core import MAX_INVALID, play_match
from urn_exchange.schedule import Config, load_config
from urn_exchange.providers import FakeProvider, Provider, ProviderResponse, ToolCallOut, Usage

ROOT = Path(__file__).resolve().parents[1]


def act_call(args, cid=None, name="act"):
    return {"name": name, "arguments": args, **({"id": cid} if cid else {})}


def guess_policy(guess=3):
    return lambda messages, tools: {"tool_calls": [act_call({"guess": guess, "belief": {"p": 0.5}})]}


def ctx(match="m1", seat_run="run-a", seed=0, deal="pilot-01"):
    return R.MatchContext(seat_run, match, "toy_duel", "0.0.1", deal, seed)


def make_player(provider, game, seat=0, **kw):
    return R.ModelPlayer(f"model{seat}", provider, game, seat, kw.pop("context", ctx()), **kw)


def bot(name, game, seat):
    return BOTS[name](game, seat)


# --------------------------------------------------------------------------- ModelPlayer


def test_message_shape_tool_results_and_system():
    g = ToyDuel()
    seen: list[list[dict]] = []

    def policy(messages, tools):
        seen.append([dict(m) for m in messages])
        assert [t["name"] for t in tools] == ["act"] and tools[0]["parameters"]["required"] == ["guess", "belief"]
        return {"tool_calls": [act_call({"guess": 4, "belief": {"p": 0.2}}, cid=f"c{len(seen)}")]}

    p = make_player(FakeProvider(policy=policy), g)
    log = play_match(g, [p, bot("near", g, 1)], "pilot-01")
    assert log.invalid == [0, 0]
    assert p.system.startswith("You are playing DeskBench Games") and "seat 0 of 2" in p.system  # v1 wire text
    assert p.system.endswith(g.rules(0))
    first, second = seen
    assert len(first) == 1 and first[0]["role"] == "user"
    assert first[0]["content"] == R.view_content(R.View("Round 1. Your turn.", {"round": 1, "seat": 0}))
    # the second view answers the first act call as its tool result
    assert second[1]["role"] == "assistant" and second[1]["tool_calls"][0]["id"] == "c1"
    assert second[2] == {"role": "tool", "tool_call_id": "c1", "name": "act",
                         "content": R.view_content(R.View("Round 2. Your turn.", {"round": 2, "seat": 0}))}
    assert len(p.calls) == 2 and all(c["kind"] == "act" for c in p.calls)


def test_nudge_on_text_only_then_empty_action():
    g = ToyDuel()
    script = [{"text": "I think 4."}, {"tool_calls": [act_call({"guess": 4, "belief": {}})]}]
    p = make_player(FakeProvider(script=script), g)
    g.reset("pilot-01", 0, 2)
    assert p.act(g.view(0), g.action_schema(0)) == {"guess": 4, "belief": {}}
    assert [c["kind"] for c in p.calls] == ["act", "nudge"]
    assert p.messages[2] == {"role": "user", "content": R.NUDGE}

    g2 = ToyDuel()
    mute = make_player(FakeProvider(policy=lambda m, t: {"text": "hmm"}), g2)
    log = play_match(g2, [mute, bot("near", g2, 1)], "pilot-01")
    # every act gives up after one nudge; the game rejects {} and applies its default after MAX_INVALID
    assert log.invalid[0] == 2 * MAX_INVALID and log.defaulted[0] == 2
    assert len(mute.calls) == 2 * MAX_INVALID * 2
    assert all(m["role"] in ("user", "assistant") for m in mute.messages)
    roles = [m["role"] for m in mute.messages]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # strict alternation


def test_wrong_tool_name_and_parallel_calls_are_all_answered():
    g = ToyDuel()
    script = [{"tool_calls": [act_call({}, "w1", name="submit")]},
              {"tool_calls": [act_call({"guess": 1, "belief": {}}, "a1"), act_call({"guess": 9, "belief": {}}, "a2")]},
              {"tool_calls": [act_call({"guess": 2, "belief": {}}, "a3")]}]
    p = make_player(FakeProvider(script=script), g)
    log = play_match(g, [p, bot("near", g, 1)], "pilot-01")
    assert [s["action"]["guess"] for s in log.steps if s["seat"] == 0] == [1, 2]
    tool_msgs = [m for m in p.messages if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "w1" and "Unknown tool 'submit'" in tool_msgs[0]["content"]
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert by_id["a2"] == R.IGNORED_RESULT and "Round 2" in by_id["a1"]


def test_error_feedback_round_trip():
    g = ToyDuel()
    script = [{"tool_calls": [act_call({"guess": 42, "belief": {}}, "x1")]},
              {"tool_calls": [act_call({"guess": 5, "belief": {}}, "x2")]},
              {"tool_calls": [act_call({"guess": 5, "belief": {}}, "x3")]}]
    p = make_player(FakeProvider(script=script), g)
    log = play_match(g, [p, bot("near", g, 1)], "pilot-01")
    assert log.invalid == [1, 0] and p.errors == ["guess must be 0-9"]
    answer = next(m for m in p.messages if m.get("tool_call_id") == "x1")
    assert answer["content"].startswith("ERROR: guess must be 0-9")


class StatefulFake(Provider):
    """Returns reasoning state and asserts it comes back unmodified on the next request."""

    provider_name = "statefake"
    wire_version = "7-test"

    def __init__(self):
        super().__init__("fake-model")
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools, system=None, params=None):
        self.requests.append([dict(m) for m in messages])
        n = sum(m["role"] == "assistant" for m in messages)
        state = {"openrouter_reasoning_details": [{"type": "reasoning.encrypted", "data": f"blob-{n}", "id": None}]}
        return ProviderResponse(text="", tool_calls=[ToolCallOut(f"t{n}", "act", {"guess": 1, "belief": {}})],
                                usage=Usage(100, 50, 10, 0), cost_usd=0.001, latency_s=0.01,
                                raw_request_hash=self.request_hash(messages, tools, system, params), model="m",
                                stop_reason="length" if n == 0 else "tool_calls", provider_state=state,
                                billed_cost_usd=0.002, upstream_provider="host-x")


def test_provider_state_echo_and_call_records():
    g = ToyDuel()
    prov = StatefulFake()
    p = make_player(prov, g, params={"max_tokens": 16000, "reasoning_effort": "high"})
    play_match(g, [p, bot("near", g, 1)], "pilot-01")
    second = prov.requests[1]
    assert second[1]["provider_state"] == {"openrouter_reasoning_details": [
        {"type": "reasoning.encrypted", "data": "blob-0", "id": None}]}
    c0 = p.calls[0]
    for k in ("request_hash", "usage", "cost_usd", "latency_s", "stop_reason", "truncated", "billed_cost_usd",
              "upstream_provider", "provider_state", "est_input_tokens", "cache_hit"):
        assert k in c0, k
    assert c0["truncated"] is True and p.calls[1]["truncated"] is False
    s = p.summary()
    assert s["calls"] == 2 and s["truncated"] == 1 and s["cached_input_tokens"] == 20
    json.dumps(p.calls)  # serialisable


def test_cache_keys_differ_across_seats_mirrors_seeds_runs():
    prov = FakeProvider(policy=guess_policy())
    g = ToyDuel()
    msgs = [{"role": "user", "content": "same"}]
    tools = [R.act_tool({"type": "object"})]

    def key(seat=0, **c):
        p = R.ModelPlayer("k", prov, g, seat, ctx(**c), params={"max_tokens": 16000})
        return prov.request_hash(msgs, tools, "sys", p.key_params())

    base = key()
    assert len({base, key(seat=1), key(match="m1-mirror"), key(seed=1), key(seat_run="run-b"),
                key(deal="pilot-02")}) == 6
    assert key() == base
    # the provider wire version is part of the key too
    class OtherWire(StatefulFake):
        wire_version = "8-test"

    assert StatefulFake().request_hash(msgs, tools, "sys", {"a": 1}) != OtherWire().request_hash(msgs, tools, "sys",
                                                                                               {"a": 1})


def test_cache_replays_only_same_run(tmp_path):
    from urn_exchange.cache import ResponseCache

    cache = ResponseCache(tmp_path / "cache")

    def play(run_id):
        g = ToyDuel()
        prov = FakeProvider(policy=guess_policy())
        p = R.ModelPlayer("k", prov, g, 0, ctx(seat_run=run_id), cache=cache)
        play_match(g, [p, bot("near", g, 1)], "pilot-01")
        return p, prov

    p1, _ = play("run-a")
    p2, prov2 = play("run-a")
    p3, _ = play("run-b")
    assert not any(c["cache_hit"] for c in p1.calls)
    assert all(c["cache_hit"] for c in p2.calls) and prov2.calls == 0
    assert not any(c["cache_hit"] for c in p3.calls)


# --------------------------------------------------------------------------- tournaments


def toy_cfg(tmp_path, models=("a", "b", "c"), games=None, **kw) -> Config:
    data = {"name": "toy", "budget_usd": kw.pop("budget_usd", 100.0),
            "models": [{"key": m, "provider": "fake", "model": "fake-model"} for m in models],
            "games": games or [{"game": "toy_duel", "n_deals": 2}, {"game": "toy_solo", "n_deals": 2},
                               {"game": "toy_pit", "n_deals": 2, "n_players": 3}],
            "ledger": str(tmp_path / "ledger.sqlite"), "results_dir": str(tmp_path / "games"),
            "cache_dir": str(tmp_path / "cache"), "concurrency": 3, **kw}
    return Config.model_validate(data)


def provs(models, **kw):
    return {m: FakeProvider(policy=guess_policy(i), **kw) for i, m in enumerate(models)}


def test_live_fake_run_writes_records_and_manifest(tmp_path):
    cfg = toy_cfg(tmp_path)
    out = R.run_tournament(cfg, providers=provs(["a", "b", "c"]), registry=toy_registry(), run_id="t1")
    assert out["status"] == "finished" and out["error"] == 0
    assert out["ok"] == out["matches"] == 3 * 2 * 2 + 3 * 2 + 2 * 3  # duel pairs, solo, pit 1 group x 3 rotations
    run_dir = Path(out["run_dir"])
    recs = [json.loads(line) for line in (run_dir / "matches.jsonl").read_text().splitlines()]
    r = next(x for x in recs if x["game_id"] == "toy_duel")
    assert r["status"] == "ok" and r["players"][0]["calls"] and r["steps"] and r["result"]["payoffs"]
    assert r["games_wire"] == R.GAMES_WIRE and r["players"][0]["system"].startswith("You are playing")
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["summary"]["ok"] == out["ok"] and man["games"]["toy_solo"]["headline"] == {"err": "lower_is_better"}
    events = (run_dir / "events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["event"] == "run_start" and json.loads(events[-1])["event"] == "run_end"


def test_dry_run_projection_uses_roster_prices(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a", "b"), budget_usd=0.0, estimate_output_tokens=1000)
    cfg.models[0].model = "openai/gpt-6.2-sol"
    cfg.models[1].model = "google/gemini-4-argon"
    cfg.models[1].estimate = {"input_tokens_per_call": 10000, "cached_fraction": 0.5, "output_tokens_per_call": 900}
    out = R.run_tournament(cfg, dry_run=True, registry=toy_registry(), run_id="dry1")
    assert out["ok"] == out["matches"] and out["dry_run"]
    pr = out["estimate"]
    a, b = pr["per_model"]["a"], pr["per_model"]["b"]
    assert a["calls"] > 0 and pr["expected_usd"] > 0 and pr["worst_case_usd"] > pr["expected_usd"]
    assert a["estimate"] == "flat" and b["estimate"] == "v1 profile"
    assert b["expected_usd"] == pytest.approx(b["calls"] * (10000 * (0.5 * 2.0 + 0.5 * 0.1) + 900 * 10.0) / 1e6, abs=1e-4)
    assert out["guard_worst_case_usd"] > 0 and out["calls_per_seat"] > 0
    rec = json.loads((Path(out["run_dir"]) / "matches.jsonl").read_text().splitlines()[0])
    assert rec["players"][0].get("dry_run_bot") in BOTS


def test_resume_skips_finished_matches(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a", "b"), games=[{"game": "toy_duel", "n_deals": 3}], circuit_breaker=99)
    broken = provs(["a", "b"])
    broken["b"] = FakeProvider(policy=guess_policy(), fault="bad_request")
    out = R.run_tournament(cfg, providers=broken, registry=toy_registry(), run_id="res1")
    assert out["error"] == out["matches"] == 6  # b plays in every match
    healthy = provs(["a", "b"])
    out2 = R.run_tournament(cfg, providers=healthy, registry=toy_registry(), resume="res1")
    assert out2["played"] == 6 and out2["ok"] == 6
    calls_before = healthy["a"].calls
    out3 = R.run_tournament(cfg, providers=healthy, registry=toy_registry(), resume="res1")
    assert out3["played"] == 0 and out3["skipped_resume"] == 6 and healthy["a"].calls == calls_before
    latest = R.latest_by_match(R.load_records(Path(out["run_dir"])))
    assert all(r["status"] == "ok" for r in latest.values())


def test_circuit_breaker_trips_a_failing_model(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a", "b", "c"), games=[{"game": "toy_solo", "n_deals": 4}], concurrency=1,
                  circuit_breaker=2)
    p = provs(["a", "b", "c"])
    p["b"] = FakeProvider(policy=guess_policy(), fault="bad_request")
    out = R.run_tournament(cfg, providers=p, registry=toy_registry(), run_id="cb1")
    assert out["tripped"] == {"b": "bad_request"}
    assert out["error"] == 2 and out["skipped"] == 2 and out["ok"] == 8 and out["status"] == "finished"


def test_fatal_error_halts_everything(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a", "b"), games=[{"game": "toy_solo", "n_deals": 4}], concurrency=1)
    p = provs(["a", "b"])
    p["a"] = FakeProvider(policy=guess_policy(), fault="auth")
    out = R.run_tournament(cfg, providers=p, registry=toy_registry(), run_id="fatal1")
    assert out["status"] == "halted" and out["halt"]["reason"] == "auth"
    assert out["error"] == 1 and out["skipped"] == 7


def test_external_stop_event(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a",), games=[{"game": "toy_solo", "n_deals": 4}])
    ev = threading.Event()
    ev.set()
    out = R.run_tournament(cfg, providers=provs(["a"]), registry=toy_registry(), stop_event=ev, run_id="sig")
    assert out["skipped"] == 4 and out["halt"] == {"reason": "signal"}


def test_only_wrong_tools_after_nudge_view_rides_on_tool_result():
    g = ToyDuel()
    wrong = {"tool_calls": [act_call({}, None, name="submit")]}
    right = {"tool_calls": [act_call({"guess": 3, "belief": {}})]}
    calls = {"n": 0}

    def policy(messages, tools):
        calls["n"] += 1
        return wrong if calls["n"] <= 2 else right

    p = make_player(FakeProvider(policy=policy), g)
    log = play_match(g, [p, bot("near", g, 1)], "pilot-01")
    assert log.invalid[0] == 1  # the first act returned {}
    roles = [m["role"] for m in p.messages]
    assert not any(a == b == "user" for a, b in zip(roles, roles[1:]))
    carrier = [m for m in p.messages if m["role"] == "tool"][1]
    assert carrier["content"].startswith("Unknown tool 'submit'") and "ERROR:" in carrier["content"]


def test_paid_run_refuses_without_confirmation_or_budget(tmp_path):
    paid = lambda: provs(["a", "b"], paid=True)  # noqa: E731
    cfg = toy_cfg(tmp_path, models=("a", "b"), budget_usd=5.0)
    with pytest.raises(R.SpendNotConfirmed, match="confirm"):
        R.run_tournament(cfg, providers=paid(), registry=toy_registry())
    cfg0 = toy_cfg(tmp_path, models=("a", "b"), budget_usd=0.0)
    with pytest.raises(R.SpendNotConfirmed, match="budget_usd"):
        R.run_tournament(cfg0, providers=paid(), registry=toy_registry(), confirmed=True)
    assert not (tmp_path / "games").exists()  # refused before any directory, ledger or call
    out = R.run_tournament(cfg, providers=paid(), registry=toy_registry(), confirmed=True, run_id="paid1")
    assert out["ok"] == out["matches"] and out["run_spent_usd"] == 0.0  # fake-model is priced at 0


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    class Flaky(FakeProvider):
        def complete(self, messages, tools, system=None, params=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                from urn_exchange.providers import ProviderError
                raise ProviderError("gemini_api/x: RemoteProtocolError: Server disconnected without sending a response.")
            return super().complete(messages, tools, system, params)

    prov = Flaky(policy=guess_policy())
    resp = R.complete_with_backoff(prov, [{"role": "user", "content": "x"}], [], None, {}, sleep=lambda s: None)
    assert calls["n"] == 3 and resp.tool_calls
    bad = FakeProvider(policy=guess_policy(), fault="bad_request")
    with pytest.raises(Exception):
        R.complete_with_backoff(bad, [{"role": "user", "content": "x"}], [], None, {}, sleep=lambda s: None)
    assert bad.calls == 1  # a bad request is never retried


def test_v1_config_is_valid_fail_safe_and_matches_the_published_run():
    cfg = load_config(ROOT / "configs" / "v1.yaml")
    assert cfg.budget_usd == 0  # nobody spends by accident: a paid run refuses until it is set
    assert cfg.model_keys() == ["gpt-6.2-astra", "gpt-6.2-sol", "gemini-4-argon", "gemini-4-carbon",
                                "claude-fable-5.2", "claude-sonnet-5.5"]
    assert cfg.params == {"max_tokens": 16000, "reasoning_effort": "high"} and cfg.concurrency == 24
    eff = {m.key: m for m in cfg.effective_models()}
    for k in ("gemini-4-argon", "gemini-4-carbon"):
        assert eff[k].params["reasoning"] == {"max_tokens": 8000}
    assert all(m.routing and m.routing.get("allow_fallbacks") is False and m.rpm and m.estimate for m in cfg.models)
    assert all(m.api_key_env == "OPENROUTER_API_KEY" and m.provider == "openai_compatible" for m in cfg.models)
    from urn_exchange.pricing import get_price

    for m in cfg.models:
        assert get_price(m.model, m.price_override).verified, m.model
    (g,) = cfg.games
    assert (g.game, g.split, g.n_deals, g.seeds, g.n_players) == ("exchange", "full", 6, [0, 1], 6)
    sched = R.build_schedule(cfg, R.catalog_from(R.default_registry()[0]))
    assert len(sched) == 72
    # the published run's match id for deal full-02 seed 0, rotation 1
    assert "exchange/full-02/s0/g0/r1/gpt-6.2-sol+gemini-4-argon+gemini-4-carbon+claude-fable-5.2+" \
           "claude-sonnet-5.5+gpt-6.2-astra" in {m.match_id for m in sched}
    for name in ("smoke.yaml", "v1_xhigh.yaml"):
        c = load_config(ROOT / "configs" / name)
        assert c.budget_usd == 0 and all(m.estimate for m in c.models)
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert len(R.build_schedule(smoke, R.catalog_from(R.default_registry()[0]))) == 1


def test_smoke_config_dry_run_end_to_end(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.ledger, cfg.results_dir = str(tmp_path / "l.sqlite"), str(tmp_path / "runs")
    out = R.run_tournament(cfg, dry_run=True, run_id="smoke-dry")
    assert out["matches"] == out["ok"] == 1 and out["error"] == 0
    assert 0.05 < out["estimate"]["expected_usd"] < 3.0
    rec = json.loads((Path(out["run_dir"]) / "matches.jsonl").read_text().splitlines()[0])
    assert rec["players"][0]["system"].startswith(R.ARENA_PREAMBLE.split("{")[0])
    assert "URN EXCHANGE" in rec["players"][0]["system"]


def test_spend_cap_halts_the_run(tmp_path):
    cfg = toy_cfg(tmp_path, models=("a", "b"), budget_usd=0.05, concurrency=1,
                  games=[{"game": "toy_duel", "n_deals": 4}])
    pricey = {m: FakeProvider(policy=guess_policy(), paid=True,
                              price={"input_per_m": 10.0, "output_per_m": 50.0}) for m in ("a", "b")}
    out = R.run_tournament(cfg, providers=pricey, registry=toy_registry(), confirmed=True, run_id="cap1")
    assert out["status"] == "halted" and out["halt"]["reason"] == "spend_cap"
    assert out["run_spent_usd"] <= 0.05 + 1e-9
    assert out["skipped"] > 0
    # a resume of the same run id keeps counting what it already spent
    out2 = R.run_tournament(cfg, providers=pricey, registry=toy_registry(), confirmed=True, resume="cap1")
    assert out2["status"] == "halted" and out2["run_spent_usd"] <= 0.05 + 1e-9
