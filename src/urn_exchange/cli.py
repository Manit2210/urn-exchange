"""``urn-exchange`` command line.

  urn-exchange dry-run  --config configs/v1.yaml                 free: scripted bots play every model seat
  urn-exchange schedule --config configs/v1.yaml [--matches]     match counts, no calls
  urn-exchange run      --config configs/smoke.yaml [--yes] [--resume RUN_ID]      PAID: calls the models
  urn-exchange ratings  --run results/runs/<run_id> [--out DIR] [--dataset v1]
  urn-exchange traces   --run results/runs/<run_id> --out traces/my-run.jsonl.gz
  urn-exchange play     [--deal pilot-01] [--seed 0] [--bots bayes,follower] [--seat 1]

Keys come from environment variables only (see .env.example). A paid run needs a non-zero ``budget_usd``
in the config and a confirmation (``--yes`` or typing ``yes`` at the prompt).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from urn_exchange.util import redact


def _print(obj) -> None:
    print(redact(json.dumps(obj, indent=2, sort_keys=True, default=str)))


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _load(args):
    from urn_exchange.schedule import load_config

    cfg = load_config(args.config)
    if getattr(args, "results_dir", None):
        cfg.results_dir = args.results_dir
    if getattr(args, "ledger", None):
        cfg.ledger = args.ledger
    return cfg


def cmd_schedule(args) -> int:
    from urn_exchange.runner import catalog_from, default_registry
    from urn_exchange.schedule import build_schedule, summarize

    cfg = _load(args)
    found, _ = default_registry()
    sched = build_schedule(cfg, catalog_from(found), models=_csv(args.models))
    out = summarize(sched).to_dict()
    if args.matches:
        out["match_ids"] = [m.match_id for m in sched]
    _print(out)
    return 0


def _estimate_lines(est: dict) -> list[str]:
    lines = [f"{'model':24s} {'matches':>7s} {'calls':>6s} {'expected USD':>13s} {'worst case USD':>15s}  basis"]
    for k, a in sorted(est["per_model"].items()):
        lines.append(f"{k:24s} {a['matches']:7d} {a['calls']:6d} {a['expected_usd']:13.2f} {a['worst_case_usd']:15.2f}"
                     f"  {a['estimate']}")
    lines.append(f"{'total':24s} {'':7s} {'':6s} {est['expected_usd']:13.2f} {est['worst_case_usd']:15.2f}")
    lines.append(f"budget_usd in the config: {est['budget_usd']:.2f}")
    return lines


def cmd_dry_run(args) -> int:
    from urn_exchange.runner import run_tournament

    cfg = _load(args)
    out = run_tournament(cfg, dry_run=True, run_id=args.run_id, models=_csv(args.models),
                         progress=None if args.quiet else (lambda s: print(s, file=sys.stderr)))
    print(f"\nDry run {out['run_id']}: {out['matches']} matches, {out['ok']} ok, {out['error']} errors "
          f"({out['wall_s']:.1f}s). Output: {out['run_dir']}")
    from urn_exchange.runner import load_records

    bots = {}
    for r in load_records(out["run_dir"]):
        for p in r.get("players", []):
            if p.get("dry_run_bot"):
                bots[p["model_key"]] = p["dry_run_bot"]
    print("Scripted bots played the model seats (results are the bots', under the model names): "
          + ", ".join(f"{k}={v}" for k, v in sorted(bots.items())))
    print("Estimated cost of the paid run (no calls were made):")
    for line in _estimate_lines(out["estimate"]):
        print("  " + line)
    print("Expected uses measured v1 token profiles where the config has one; worst case assumes no cache and "
          "every reply at max_tokens.")
    return 0 if out["error"] == 0 else 1


def cmd_run(args) -> int:
    from urn_exchange.runner import SpendNotConfirmed, run_tournament

    cfg = _load(args)
    print(f"PAID RUN: {len(cfg.models)} models, budget_usd {cfg.budget_usd:.2f} (hard ceiling for this run id). "
          "Run `urn-exchange dry-run` first for an estimate.", file=sys.stderr)
    confirmed = args.yes
    if not confirmed and cfg.budget_usd > 0:
        if not sys.stdin.isatty():
            print("refusing: pass --yes to confirm the spend in a non-interactive shell", file=sys.stderr)
            return 2
        confirmed = input("Type yes to start calling paid models: ").strip().lower() == "yes"
        if not confirmed:
            print("not started", file=sys.stderr)
            return 2
    stop = threading.Event()

    def on_signal(signum, frame):  # finish in-flight matches, start no new ones; resume later
        print("\nstopping after in-flight matches (resume with --resume)", file=sys.stderr)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        out = run_tournament(cfg, confirmed=confirmed, run_id=args.run_id, resume=args.resume,
                             models=_csv(args.models), stop_event=stop, events_stdout=args.events_stdout,
                             progress=lambda s: print(s, file=sys.stderr))
    except SpendNotConfirmed as e:
        print(str(e), file=sys.stderr)
        return 2
    _print(out)
    if out["status"] == "halted" or out["error"] or out["skipped"]:
        print(f"Not complete: resume with `urn-exchange run --config {args.config} --resume {out['run_id']}`",
              file=sys.stderr)
    return 3 if out.get("halt") else 0


def cmd_ratings(args) -> int:
    from urn_exchange.ratings import compute, load_runs, write_outputs
    from urn_exchange.summary import summarize, write_match_csv

    sources = _csv(args.run) or []
    records, meta = load_runs(sources)
    datasets = sorted({r.get("dataset", "") for r in records})
    if args.dataset:
        records = [r for r in records if r.get("dataset", "") == args.dataset]
    elif len(datasets) > 1:
        print(f"the input holds several datasets {datasets}; choose one with --dataset", file=sys.stderr)
        return 2
    if not records:
        print("no match records found", file=sys.stderr)
        return 1
    first = Path(sources[0])
    out = Path(args.out) if args.out else (first / "report" if first.is_dir() else first.parent / "report")
    res = compute(records, meta, n_boot=args.n_boot, cluster=args.cluster)
    res["sources"] = [Path(s).name for s in sources]
    files = write_outputs(res, out)
    n = write_match_csv(records, out / "matches.csv")
    summ = summarize([r for r in records if r.get("game_id") == "exchange"], res)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print((out / "report.md").read_text())
    print(f"Wrote {', '.join(sorted(Path(f).name for f in files.values()))}, matches.csv ({n} rows), summary.json "
          f"to {out}")
    return 0


def cmd_traces(args) -> int:
    from urn_exchange.traces import export

    sources = [(s, args.dataset or Path(s).name) for s in (_csv(args.run) or [])]
    _print(export(sources, args.out))
    return 0


def cmd_play(args) -> int:
    from urn_exchange.play import play

    play(args.deal, args.seed, _csv(args.bots), args.seat)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="urn-exchange", description="Urn Exchange: a zero-sum trading game for "
                                 "language models (paper chips, research only, not investment advice).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dry-run", help="free: every model seat played by a scripted bot; prints a cost estimate")
    d.add_argument("--config", required=True)
    d.add_argument("--run-id", default=None)
    d.add_argument("--models", default=None, help="comma-separated model keys (a slice of the roster)")
    d.add_argument("--results-dir", default=None)
    d.add_argument("--ledger", default=None)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_dry_run)

    s = sub.add_parser("schedule", help="match counts of a config (no calls)")
    s.add_argument("--config", required=True)
    s.add_argument("--models", default=None)
    s.add_argument("--matches", action="store_true", help="also print every match id")
    s.set_defaults(func=cmd_schedule)

    r = sub.add_parser("run", help="PAID: play the tournament with real models")
    r.add_argument("--config", required=True)
    r.add_argument("--yes", action="store_true", help="confirm the spend without a prompt")
    r.add_argument("--run-id", default=None)
    r.add_argument("--resume", default=None, help="run id to resume (skips matches already finished ok)")
    r.add_argument("--models", default=None, help="comma-separated model keys (a slice of the roster)")
    r.add_argument("--results-dir", default=None)
    r.add_argument("--ledger", default=None)
    r.add_argument("--events-stdout", action="store_true")
    r.set_defaults(func=cmd_run)

    ra = sub.add_parser("ratings", help="chips, BT Elo with bootstrap ranges, skill profile, per-match CSV")
    ra.add_argument("--run", required=True, help="run directory or trace file (comma-separated for several)")
    ra.add_argument("--out", default=None, help="output directory (default: <run>/report)")
    ra.add_argument("--dataset", default=None, help="for trace files with several datasets, e.g. v1 or v1-xhigh")
    ra.add_argument("--n-boot", type=int, default=1000)
    ra.add_argument("--cluster", choices=["unit", "deal"], default="unit")
    ra.set_defaults(func=cmd_ratings)

    t = sub.add_parser("traces", help="export a run as a shareable .jsonl.gz without encrypted reasoning blobs")
    t.add_argument("--run", required=True, help="run directory (comma-separated for several)")
    t.add_argument("--out", required=True)
    t.add_argument("--dataset", default=None, help="label stored on every record (default: the run id)")
    t.set_defaults(func=cmd_traces)

    p = sub.add_parser("play", help="play a match yourself against scripted bots (free)")
    p.add_argument("--deal", default="pilot-01")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bots", default="bayes,follower", help="comma-separated bots for the other seats (2-5): "
                   "random, alwayspass, bayes, halfkelly, noisy, follower, pooled")
    p.add_argument("--seat", type=int, default=1, help="your seat, 1-based")
    p.set_defaults(func=cmd_play)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
