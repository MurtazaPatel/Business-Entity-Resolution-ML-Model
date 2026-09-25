"""Single entry point for every pipeline step.

    python code/business_entity_resolution/src/run_step.py --step p0_load

Each step is a callable in STEPS returning markdown lines; the runner wraps it with
runtime and peak RAM and writes reports/<id>.md.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Paths, add_common_args, paths_from_args, set_seed  # noqa: E402
import io_utils  # noqa: E402
import eda  # noqa: E402

REPORT_MAX_LINES = 200


def peak_ram_gb() -> float:
    """maxrss is bytes on macOS and kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2


def md_table(df, max_cell: int = 60) -> str:
    """Small markdown table. Hand-rolled so reports never depend on `tabulate`."""
    def cell(v: str) -> str:
        v = "" if v is None else str(v)
        v = v.replace("|", "\\|").replace("\n", " ")
        return (v[: max_cell - 1] + "\u2026") if len(v) > max_cell else v

    cols = [cell(c) for c in df.columns]
    rows = [[cell(v) for v in row] for row in df.astype(str).itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def step_p1a_check(paths: Paths, args: argparse.Namespace) -> list[str]:
    """Shape, columns and 3 sample rows for each of the 7 given files."""
    lines: list[str] = []
    for name, path in paths.all_sources().items():
        if not path.exists():
            lines.append(f"### {name}\n\n**MISSING**: `{path}`\n")
            continue
        n_rows = io_utils.count_rows(path)
        head = io_utils.peek(path, 3)
        size_mb = path.stat().st_size / 1024**2

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- shape: **({n_rows:,}, {head.shape[1]})**   size: {size_mb:,.1f} MB")
        lines.append(f"- columns: `{list(head.columns)}`")
        lines.append("")
        lines.append(md_table(head))
        lines.append("")

    if args.sample:
        sample = io_utils.load_sample(paths, args.sample, seed=args.seed)
        lines.append(f"### sample slice (--sample {args.sample})")
        lines.append("")
        for key, frame in sample.items():
            lines.append(f"- `{key}`: {frame.shape}")
        lines.append("")
    return lines


def step_p1_eda(paths: Paths, args: argparse.Namespace) -> list[str]:
    """Dataset facts that decide the blocking and matching design."""
    return eda.run(paths, args)


# Canonical ids are p1a_check / p1b_eda. Older ids stay as aliases so a notebook or
# report already pointing at one keeps working.
STEPS = {
    "p1a_check": step_p1a_check,
    "p1b_eda": step_p1_eda,
    # aliases
    "p0_load": step_p1a_check,
    "p1_eda": step_p1_eda,
}
CANONICAL = ("p1a_check", "p1b_eda")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline step.")
    parser.add_argument(
        "--step",
        required=True,
        metavar="ID",
        help=f"Step id to run. Available: {', '.join(CANONICAL)}.",
    )
    parser.add_argument("--no-report", action="store_true", help="Print only; skip reports/<id>.md.")
    parser.add_argument("--probe", type=int, default=200_000,
                        help="Sample size for the per-pair blocking probes (p1_eda).")
    add_common_args(parser)
    args = parser.parse_args()

    if args.step not in STEPS:
        import difflib

        # Prefer a canonical id over an alias when suggesting.
        near = difflib.get_close_matches(args.step, CANONICAL, n=1, cutoff=0.4) or \
            difflib.get_close_matches(args.step, STEPS, n=1, cutoff=0.5)
        hint = f"  Did you mean '{near[0]}'?" if near else ""
        parser.error(
            f"unknown step '{args.step}'.{hint}\n"
            f"  Available: {', '.join(CANONICAL)}"
        )

    set_seed(args.seed)
    paths = paths_from_args(args)
    paths.ensure_dirs()

    started = time.perf_counter()
    body = STEPS[args.step](paths, args)
    elapsed = time.perf_counter() - started

    header = [
        f"# Step `{args.step}`",
        "",
        f"- date: {date.today().isoformat()}",
        f"- runtime: {elapsed:.1f}s   peak RAM: {peak_ram_gb():.2f} GB",
        f"- data-dir: `{Path(args.data_dir).relative_to(paths.repo_root) if Path(args.data_dir).is_relative_to(paths.repo_root) else args.data_dir}`"
        f"   sample: {args.sample or 'full'}",
        "",
    ]
    report = header + body
    if len(report) > REPORT_MAX_LINES:
        report = report[:REPORT_MAX_LINES] + ["", f"_(truncated at {REPORT_MAX_LINES} lines)_"]

    text = "\n".join(report) + "\n"
    print(text)

    if not args.no_report:
        out = paths.reports / f"{args.step}.md"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(paths.repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
