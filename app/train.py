"""Train the local routing model from the request log.

    python -m app.train              # train, evaluate on held-out traffic, save
    python -m app.train --dry-run    # report only, write nothing
    python -m app.train --json       # machine-readable report (for cron/CI)

The server only routes with a model whose held-out replay showed it beating
the static tiers. A retrain that fails that check never replaces one that
passed it: the candidate goes to `<path>.candidate.json` instead, so a bad
week of traffic can't knock out a working router. A running server picks up a
new model within ~30 seconds — no restart needed.
"""

import argparse
import asyncio
import json
import os
import sys

from .config import load_config
from .store import Store
from .predictor import RouterModel
from .training import NotEnoughData, should_replace, train


def _money(value) -> str:
    return "—" if value is None else f"${value:.6f}"


def _print_report(report, saved_to: str | None, note: str | None) -> None:
    print(f"Examples: {report.examples} requests "
          f"({report.train_examples} train / {report.test_examples} held out, newest last)")
    print(f"Heads: thresholds {report.thresholds} × {len(report.model.ladder)} rungs\n")

    print("Held-out quality per rung (observed outcomes only):")
    print(f"  {'model':<34} {'n':>5} {'pass':>6} {'auc':>6} {'logloss':>8} {'base':>8}")
    fmt = lambda v, spec: "—" if v is None else format(v, spec)  # noqa: E731
    for m in report.head_metrics:
        if m.get("rung") == "all":
            p = m["pooled"]
            print(f"  {'all rungs, pooled':<34} {p['n']:>5} {'':>6} {'':>6} "
                  f"{fmt(p['log_loss'], '.4f'):>8} {fmt(p['baseline_log_loss'], '.4f'):>8}")
            continue
        print(f"  {m['model']:<34} {m['test_n']:>5} {fmt(m['test_pass_rate'], '.3f'):>6} "
              f"{fmt(m['auc'], '.3f'):>6} {fmt(m['log_loss'], '.4f'):>8} "
              f"{fmt(m['baseline_log_loss'], '.4f'):>8}")
    print("  (log loss below 'base' means the model learned something about prompts)\n")

    replay = report.replay
    print(f"Cost replay on {replay['covered']}/{replay['requests']} held-out requests, "
          f"threshold {replay['threshold']}")
    print("  unknown outcomes are filled with the model's predicted P(pass):")
    print(f"  {'policy':<18} {'mean cost':>12} {'success':>8} {'attempts':>9}")
    for name, p in replay["policies"].items():
        print(f"  {name:<18} {_money(p['mean_cost_usd']):>12} "
              f"{fmt(p['success_rate'], '.3f'):>8} {fmt(p['mean_attempts'], '.2f'):>9}")
    print(f"  {'oracle (bound)':<18} {_money(replay['oracle_mean_cost_usd']):>12}")
    obs = replay["observed_only"]
    print(f"\n  Observed-only reference ({obs['covered']} requests, understates the model):")
    for name in ("model", "static_tiers"):
        print(f"  {name:<18} {_money(obs[name]['mean_cost_usd']):>12} "
              f"{fmt(obs[name]['success_rate'], '.3f'):>8}")
    print()

    verdict = "DEPLOYABLE" if report.deployable else "NOT deployable"
    print(f"{verdict}: {report.deploy_reason}")
    if saved_to is None:
        print("Dry run — nothing written.")
        return
    refit = " (refit on all data after evaluation)" if report.refit else ""
    print(f"Saved to {saved_to}{refit}")
    if saved_to.endswith(".candidate.json"):
        print(f"Note: {note}. Pass --replace to install it anyway.")


async def _main(args) -> int:
    config = load_config(args.config)
    store = Store(config.db_path)
    await store.init()
    try:
        report = await train(config, store, refit=not args.no_refit, force=args.force)
    except NotEnoughData as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Not enough data: {exc}", file=sys.stderr)
        return 2
    finally:
        await store.close()

    path = args.output or config.predictor.path
    saved_to, note = None, None
    if not args.dry_run:
        existing = None
        if os.path.exists(path):
            try:
                existing = RouterModel.load(path)
            except Exception:
                existing = None  # unreadable — nothing worth protecting
        replace, note = should_replace(existing, report.model)
        if replace or args.replace:
            report.model.save(path)
            saved_to = path
        else:
            # Keep the candidate for inspection without letting it route.
            saved_to = path.removesuffix(".json") + ".candidate.json"
            report.model.save(saved_to)
    if args.json:
        print(json.dumps({
            "saved_to": saved_to,
            "replaced_model": saved_to == path,
            "note": note,
            "examples": report.examples,
            "deployable": report.deployable,
            "deploy_reason": report.deploy_reason,
            **report.model.metrics,
        }, indent=2))
    else:
        _print_report(report, saved_to, note)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default: $RATATOSK_CONFIG or config.yaml)")
    parser.add_argument("--output", help="where to write the model (default: predictor.path)")
    parser.add_argument("--dry-run", action="store_true", help="evaluate without saving")
    parser.add_argument("--no-refit", action="store_true",
                        help="save the model as evaluated, instead of refitting on all data")
    parser.add_argument("--force", action="store_true",
                        help="train even below predictor.min_examples")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite a deployed model even if the new one is not deployable")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    sys.exit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
