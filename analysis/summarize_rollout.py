"""Collect W5 rollout manifests without fabricating rollout metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path, output: Path) -> int:
    records = []
    for path in sorted(root.glob("**/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") == "frost_paper_accuracy_reuse_runtime":
            records.append({"source": str(path), "payload": payload})
    if not records:
        raise FileNotFoundError(f"No W5 rollout manifests under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/active"))
    parser.add_argument("--output", type=Path, default=Path("results/active/paper_evidence/rollout_summary.json"))
    args = parser.parse_args()
    print(f"wrote {summarize(args.root, args.output)} rollout summaries")
