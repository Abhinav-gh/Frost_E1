"""Collect generalization manifests and summaries that actually exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path, output: Path) -> int:
    records = []
    for path in sorted(root.glob("**/stage_a/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({"source": str(path), "stage": "A", "payload": payload})
    for path in sorted(root.glob("**/stage_b_mode*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({"source": str(path), "stage": "B", "payload": payload})
    if not records:
        raise FileNotFoundError(f"No generalization summaries under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/active"))
    parser.add_argument("--output", type=Path, default=Path("results/active/paper_evidence/generalization_summary.json"))
    args = parser.parse_args()
    print(f"wrote {summarize(args.root, args.output)} summaries")
