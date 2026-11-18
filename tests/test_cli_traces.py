"""CLI (dry-run, ratings, run refusal, play) and trace export, offline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import yaml

from urn_exchange.cli import main
from urn_exchange.play import parse_move, play
from urn_exchange.traces import export, long_tokens, strip_record

ROOT = Path(__file__).resolve().parents[1]


def _smoke(tmp_path, **over) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg.update(ledger=str(tmp_path / "spend.sqlite"), results_dir=str(tmp_path / "runs"),
               cache_dir=str(tmp_path / "cache"), **over)
    p = tmp_path / "smoke.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_cli_dry_run_ratings_and_paid_refusals(tmp_path, capsys):
    cfg = _smoke(tmp_path, games=[{"game": "exchange", "deals": ["pilot-02"], "seeds": [0], "n_players": 3}])
    assert main(["dry-run", "--config", str(cfg), "--run-id", "d1", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "3 matches, 3 ok, 0 errors" in out and "Estimated cost" in out and "gpt-6.2-sol=bayes" in out
    run_dir = tmp_path / "runs" / "d1"
    assert main(["ratings", "--run", str(run_dir), "--n-boot", "50"]) == 0
    rep = run_dir / "report"
    assert {"report.md", "ratings.json", "chips.csv", "matches.csv", "summary.json"} <= {p.name for p in rep.iterdir()}
    assert len((rep / "matches.csv").read_text().splitlines()) == 1 + 3 * 3
    assert json.loads((rep / "summary.json").read_text())["setup"]["matches"] == 3
    # a paid run refuses with budget 0, and without confirmation in a non-interactive shell
    assert main(["run", "--config", str(cfg)]) == 2
    assert "budget_usd" in capsys.readouterr().err
    cfg2 = _smoke(tmp_path, budget_usd=1.0)
    assert main(["run", "--config", str(cfg2)]) == 2
    assert "--yes" in capsys.readouterr().err
    # confirmed, but no key in the environment: the match errors on the missing key and the run halts
    assert main(["run", "--config", str(cfg2), "--yes", "--run-id", "nokey"]) == 3
    rec = json.loads((tmp_path / "runs" / "nokey" / "matches.jsonl").read_text().splitlines()[0])
    assert rec["status"] == "error" and rec["error_class"] == "missing_key" and "OPENROUTER_API_KEY" in rec["error"]


def test_trace_export_strips_blobs_and_keeps_readable_reasoning(tmp_path):
    rec = {"record": "match", "status": "ok", "match_id": "m1", "players": [{"calls": [
        {"text": "", "provider_state": {"openrouter_reasoning_details": [
            {"type": "reasoning.text", "text": "pooled posterior", "signature": "QUJD" * 200},
            {"type": "reasoning.summary", "summary": "short summary"},
            {"type": "reasoning.encrypted", "data": "gAAA" * 300, "format": "openai-responses-v1"}]}},
        {"text": "", "provider_state": {"gemini_content": {"role": "model", "parts": [
            {"function_call": {"name": "act", "args": {}}, "thought_signature": "EusE" * 300}]}}}]}]}
    s = strip_record(rec, "v1")
    det = s["players"][0]["calls"][0]["provider_state"]["openrouter_reasoning_details"]
    assert det[0] == {"type": "reasoning.text", "text": "pooled posterior", "signature_removed": True}
    assert det[1]["summary"] == "short summary"
    assert det[2]["data"] is None and det[2]["data_removed"] is True
    part = s["players"][0]["calls"][1]["provider_state"]["gemini_content"]["parts"][0]
    assert "thought_signature" not in part and part["thought_signature_removed"] is True
    assert not long_tokens(s) and long_tokens(rec) and s["dataset"] == "v1"
    run = tmp_path / "run"
    run.mkdir()
    lines = [rec, {**rec, "status": "error"}, {**rec, "match_id": "m2"}, {**rec, "match_id": "m3", "status": "error"}]
    (run / "matches.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    info = export([(run, "v1")], tmp_path / "t.jsonl.gz")
    # m1's last record errored -> dropped; m2 ok; m3 errored
    assert info["records"] == {"v1": 1} and info["long_tokens_left"] == 0
    got = [json.loads(x) for x in gzip.open(tmp_path / "t.jsonl.gz", "rt")]
    assert [g["match_id"] for g in got] == ["m2"]


def test_play_against_bots_with_scripted_input():
    assert parse_move("buy 3 60 0.7", False) == {"side": "buy", "size": 3, "limit": 60, "belief": {"p_a": 0.7}}
    assert parse_move("pass 0.4", False) == {"side": "pass", "size": 0, "belief": {"p_a": 0.4}}
    assert parse_move("yes 0.3", True) == {"research": True, "belief": {"p_a": 0.3}}
    assert parse_move('{"side": "pass", "size": 0, "belief": {"p_a": 0.5}}', False)["side"] == "pass"
    moves = iter(["buy 99 50 0.5"] + ["pass 0.5"] * 200)  # one illegal move first, then always pass
    printed: list[str] = []
    res = play("pilot-02", 0, ["bayes", "random"], 2, input_fn=lambda prompt: next(moves), print_fn=printed.append)
    assert sum(res["payoffs"]) == 0 and any("ERROR" in p for p in printed)
    assert any("URN EXCHANGE" in p for p in printed) and "Match over" in printed[-2]
