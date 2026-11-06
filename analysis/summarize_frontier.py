"""Aggregate Stage-B accuracy result files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summarize(root: Path, output: Path) -> int:
    rows = []
    for path in sorted(root.glob("**/stage_b_mode*/accuracy_results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source"] = str(path)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No Stage-B accuracy results under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", *[key for key in rows[0] if key != "source"]]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/active"))
    parser.add_argument("--output", type=Path, default=Path("results/active/paper_evidence/frontier_summary.csv"))
    args = parser.parse_args()
    print(f"wrote {summarize(args.root, args.output)} Stage-B rows")
