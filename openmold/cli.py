"""Command line interface."""
import argparse, shutil, sys
from pathlib import Path
from .config import ConfigError, load_config
from .pipeline import STEPS, run_all


def _check_tools(cfg):
    problems = []
    if not shutil.which(cfg["tools"]["ccx"]):
        problems.append(f"CalculiX binary '{cfg['tools']['ccx']}' not found. Install it or set tools.ccx in the config.")
    if cfg["exports"]["step"] and not shutil.which(cfg["tools"]["freecad"]):
        problems.append(f"exports.step needs '{cfg['tools']['freecad']}', which was not found.")
    return problems


def _warnings(cfg):
    if cfg["tools"]["ccx_threads"] > 1:
        return ["Warning: tools.ccx_threads > 1 can silently give wrong CalculiX results. See \"Solver threads\" in the README."]
    return []


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(prog="openmold", description="Compare injection-moulded part variants with gmsh and CalculiX.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the analysis and write the report")
    r.add_argument("config")
    r.add_argument("--parts", nargs="*", help="only these part ids")
    r.add_argument("--only", nargs="*", choices=STEPS, help="only these steps; earlier results must already exist")
    r.add_argument("--force", action="store_true", help="rerun steps even if they are up to date")
    r.add_argument("--jobs", type=int, default=1, help="parts to process in parallel")
    r.add_argument("--no-report", action="store_true")
    c = sub.add_parser("check", help="validate a config and look for the external tools")
    c.add_argument("config")
    p = sub.add_parser("report", help="rebuild figures and report from existing results")
    p.add_argument("config")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as err:
        print(err, file=sys.stderr)
        return 2
    for line in _warnings(cfg):
        print(line, file=sys.stderr)
    if args.cmd == "check":
        problems = _check_tools(cfg)
        print("\n".join(problems) if problems else f"OK: {len(cfg['parts'])} part(s), material '{cfg['material']['name']}', output {cfg['output']}")
        return 1 if problems else 0
    if args.cmd == "report":
        from . import plots, report
        out = Path(cfg["output"])
        print(report.build(cfg, out, plots.make_all(cfg, out)))
        return 0
    problems = _check_tools(cfg)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    target, failed = run_all(cfg, args.parts, args.only, args.force, args.jobs, not args.no_report)
    for pid, msg in failed.items():
        print(f"FAILED {pid}: {msg}", file=sys.stderr)
    if target:
        print(f"Report: {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
