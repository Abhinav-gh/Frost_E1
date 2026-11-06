#!/usr/bin/env python3
"""
Compact scientific Stage-B Mode-4 comparison for Frost G1.

Input: recursively discovered Stage-B summary.json files (one per seed).
Output: a small set of plots + compact CSV/Markdown tables.

Plots:
  1. accuracy_reuse_frontier.png
  2. force_error_budgets.png
  3. seed_reproducibility.png
  4. reference_age_vs_error.png (when selected_rows are available)
  5. runtime.png

Only scientifically useful metrics are retained. This deliberately does NOT
dump every JSON field into a giant comparison table.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


DEFAULT_GLOB = "**/stage_b_mode4/summary.json"
METRICS = [
    ("clean_fraction", "Clean fraction", "fraction"),
    ("force_rms_error_meV_A", "Force RMS error", "meV/Å"),
    ("force_max_error_meV_A", "Maximum force error", "meV/Å"),
    ("energy_error_eV", "Energy error", "eV"),
    ("relative_force_error", "Relative force error", ""),
    ("reference_age_mean_steps", "Mean reference age", "steps"),
    ("reference_age_max_steps", "Maximum reference age", "steps"),
    ("action_evaluation_seconds", "Action evaluation time", "s"),
]


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def get_seed(d, path):
    return str(d.get("seed", d.get("metadata", {}).get("seed", path.parent.parent.name)))


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def summarize(d):
    out = {}
    for item in d.get("summaries_by_tolerance", []):
        tol = float(item["tolerance_A"])
        out[tol] = {}
        for metric, _, _ in METRICS:
            v = item.get(metric)
            # Most metrics are nested {mean,median,std,min,max}; clean_fraction etc.
            if isinstance(v, dict):
                out[tol][metric] = v.get("mean")
                out[tol][metric + "_median"] = v.get("median")
                out[tol][metric + "_std"] = v.get("std")
            else:
                out[tol][metric] = v
    return out


def mean_sd(values):
    x = [float(v) for v in values if numeric(v)]
    if not x:
        return None, None
    return statistics.mean(x), (statistics.stdev(x) if len(x) > 1 else 0.0)


def save_csv(path, rows, headers):
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budgets", default="5,10,20",
                    help="maximum-force error budgets in meV/Å")
    args = ap.parse_args()

    files = sorted(args.root.glob(args.glob))
    if not files:
        files = sorted(p for p in args.root.glob("**/summary.json")
                       if "stage_b" in str(p).lower())
    if not files:
        raise SystemExit(f"No Stage-B summaries found under {args.root}")

    runs = []
    for p in files:
        d = read_json(p)
        runs.append({"path": p, "seed": get_seed(d, p), "raw": d, "tol": summarize(d)})
    out = (args.out or args.root / "stage_b_comparison").resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Validate run-level invariants; compact and visible.
    invariant_keys = ["model", "device", "dtype", "cache_mode", "sampling",
                      "sample_count_requested", "batched_action_chunk_size",
                      "dense_jacobian_enabled", "force_error_executed"]
    invariant_rows = []
    for k in invariant_keys:
        vals = [r["raw"].get(k, r["raw"].get("metadata", {}).get(k)) for r in runs]
        invariant_rows.append({"field": k, "values": " | ".join(f"{r['seed']}={v}" for r,v in zip(runs, vals)),
                              "consistent": len(set(map(str, vals))) == 1})

    tolerances = sorted(set().union(*(r["tol"].keys() for r in runs)))
    # 1) Accuracy–reuse frontier (each seed line; mean across seeds with SD band)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in runs:
        xs = [r["tol"][t].get("clean_fraction") for t in tolerances]
        ys = [r["tol"][t].get("force_rms_error_meV_A") for t in tolerances]
        pts = [(x,y) for x,y in zip(xs,ys) if numeric(x) and numeric(y)]
        if pts:
            ax.plot([x for x,y in pts], [y for x,y in pts], marker="o", alpha=.45,
                    label=f"seed {r['seed']}")
    mx, my, sy = [], [], []
    for t in tolerances:
        a = [r["tol"].get(t,{}).get("clean_fraction") for r in runs]
        b = [r["tol"].get(t,{}).get("force_rms_error_meV_A") for r in runs]
        am, _ = mean_sd(a); bm, bs = mean_sd(b)
        if numeric(am) and numeric(bm):
            mx.append(am); my.append(bm); sy.append(bs or 0)
    if mx:
        ax.errorbar(mx, my, yerr=sy, marker="D", linewidth=2, capsize=3,
                    label="Across-seed mean ± SD")
    ax.set_xlabel("Mean clean fraction")
    ax.set_ylabel("Mean force RMS error (meV/Å)")
    ax.set_title("Stage B: accuracy–reuse frontier")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"accuracy_reuse_frontier.png", dpi=180); plt.close(fig)

    # 2) Maximum-force budgets from top-level force_error_budgets
    budget_vals = [float(x) for x in args.budgets.split(",") if x.strip()]
    b_rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in runs:
        budgets = r["raw"].get("force_error_budgets", {})
        x, y = [], []
        for b in budget_vals:
            item = budgets.get(str(b), budgets.get(f"{b:.1f}", {}))
            clean = item.get("largest_observed_clean_fraction")
            if numeric(clean):
                x.append(b); y.append(float(clean))
                b_rows.append({"seed":r["seed"], "budget_meV_A":b,
                               "largest_observed_clean_fraction":clean,
                               "eligible_rows":item.get("eligible_rows")})
        if x: ax.plot(x,y,marker="o",label=f"seed {r['seed']}")
    ax.set_xlabel("Maximum force-error budget (meV/Å)")
    ax.set_ylabel("Largest observed clean fraction")
    ax.set_title("Observed reuse under maximum-force-error budgets")
    ax.grid(True,alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"force_error_budgets.png",dpi=180); plt.close(fig)
    save_csv(out/"force_error_budgets.csv",b_rows,
             ["seed","budget_meV_A","largest_observed_clean_fraction","eligible_rows"])

    # 3) Across-seed reproducibility: mean +/- SD per tolerance, compact four metrics
    rep_rows=[]
    chosen=["clean_fraction","force_rms_error_meV_A","force_max_error_meV_A","energy_error_eV"]
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for ax,(metric,title,unit) in zip(axes.flat,[
        ("clean_fraction","Clean fraction",""),
        ("force_rms_error_meV_A","Force RMS error","meV/Å"),
        ("force_max_error_meV_A","Maximum force error","meV/Å"),
        ("energy_error_eV","Energy error","eV")]):
        means, sds = [], []
        for t in tolerances:
            vals=[r["tol"].get(t,{}).get(metric) for r in runs]
            m,s=mean_sd(vals); means.append(m); sds.append(s)
            rep_rows.append({"tolerance_A":t,"metric":metric,"mean_across_seeds":m,
                             "seed_sample_sd":s,"n_seeds":sum(numeric(v) for v in vals)})
        good=[i for i,m in enumerate(means) if numeric(m)]
        ax.errorbar([tolerances[i] for i in good],[means[i] for i in good],
                    yerr=[sds[i] or 0 for i in good],marker="o",capsize=3)
        ax.set_title(title); ax.set_xlabel("Tolerance (Å)")
        if unit: ax.set_ylabel(unit)
        ax.grid(True,alpha=.25)
    fig.suptitle("Stage B: cross-seed mean ± sample SD",y=1.02)
    fig.tight_layout(); fig.savefig(out/"seed_reproducibility.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    save_csv(out/"seed_reproducibility.csv",rep_rows,
             ["tolerance_A","metric","mean_across_seeds","seed_sample_sd","n_seeds"])

    # 4) Per-sampled-row reference age vs force error, if rows are present.
    age_rows=[]
    for r in runs:
        for row in r["raw"].get("selected_rows",[]):
            age=row.get("reference_age_mean_steps")
            err=row.get("force_rms_error_meV_A")
            if numeric(age) and numeric(err):
                age_rows.append({"seed":r["seed"],"step":row.get("step"),
                    "tolerance_A":row.get("tolerance_A"),"reference_age_mean_steps":age,
                    "force_rms_error_meV_A":err,
                    "force_max_error_meV_A":row.get("force_max_error_meV_A"),
                    "clean_fraction":(row.get("n_clean",0)/row.get("n_edges")
                                     if numeric(row.get("n_edges")) and row.get("n_edges") else None)})
    if age_rows:
        fig,ax=plt.subplots(figsize=(7.5,5))
        for r in runs:
            q=[z for z in age_rows if z["seed"]==r["seed"]]
            ax.scatter([z["reference_age_mean_steps"] for z in q],
                       [z["force_rms_error_meV_A"] for z in q],label=f"seed {r['seed']}",alpha=.65)
        ax.set_xlabel("Mean reference age (steps)")
        ax.set_ylabel("Force RMS error (meV/Å)")
        ax.set_title("Sampled configurations: reference age vs force error")
        ax.grid(True,alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/"reference_age_vs_error.png",dpi=180);plt.close(fig)
        save_csv(out/"selected_rows_diagnostics.csv",age_rows,list(age_rows[0].keys()))

    # 5) Runtime bars, separately show action evaluation vs total Stage B runtime.
    runtime_rows=[]
    for r in runs:
        d=r["raw"]
        runtime_rows.append({"seed":r["seed"],
            "action_evaluation_seconds":d.get("action_evaluation_seconds"),
            "total_stage_b_runtime_seconds":d.get("total_stage_b_runtime_seconds"),
            "peak_cuda_allocated_bytes":d.get("peak_cuda_allocated_bytes")})
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    labels=[z["seed"] for z in runtime_rows]
    action=[z["action_evaluation_seconds"] for z in runtime_rows]
    total=[z["total_stage_b_runtime_seconds"] for z in runtime_rows]
    x=list(range(len(labels))); width=.36
    ax.bar([i-width/2 for i in x],action,width,label="Action evaluation")
    ax.bar([i+width/2 for i in x],total,width,label="Total Stage B")
    ax.set_xticks(x,labels);ax.set_xlabel("Seed");ax.set_ylabel("Seconds")
    ax.set_title("Stage-B runtime (not MD rollout speedup)")
    ax.grid(True,axis="y",alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/"runtime.png",dpi=180);plt.close(fig)
    save_csv(out/"runtime.csv",runtime_rows,list(runtime_rows[0].keys()))

    # concise report
    md=["# Frost G1 Stage-B compact comparison","",
        f"Compared **{len(runs)}** Stage-B summary files.",
        "","## Run consistency","",
        "| Field | Values | Consistent |","|---|---|:---:|"]
    for z in invariant_rows:
        md.append(f"| `{z['field']}` | {z['values']} | {'yes' if z['consistent'] else '**NO**'} |")
    md += ["","## Across-seed summary","",
           "| ε (Å) | Clean fraction mean ± SD | Force RMS mean ± SD (meV/Å) | Force max mean ± SD (meV/Å) | Energy error mean ± SD (eV) |",
           "|---:|---:|---:|---:|---:|"]
    for t in tolerances:
        cells=[]
        for metric in chosen:
            vals=[r["tol"].get(t,{}).get(metric) for r in runs]
            m,s=mean_sd(vals);cells.append(f"{m:.6g} ± {s:.3g}" if numeric(m) else "—")
        md.append(f"| {t:g} | "+" | ".join(cells)+" |")
    md += ["","## Plots","",
           "- `accuracy_reuse_frontier.png`: clean fraction versus force RMS error.",
           "- `force_error_budgets.png`: maximum observed reuse under force-error budgets.",
           "- `seed_reproducibility.png`: four key metrics across tolerances (mean ± SD).",
           "- `reference_age_vs_error.png`: sampled-row age/error scatter (if row data exists).",
           "- `runtime.png`: action time and total Stage-B time, shown separately.",
           "","**Interpretation caution:** Stage B evaluates sampled snapshots, not all 1,000 MD steps. These plots describe sampled accuracy/reuse and Stage-B compute cost; they do not establish full-trajectory speedup or long-rollout stability.",""]
    (out/"comparison.md").write_text("\\n".join(md),encoding="utf-8")
    print(f"Processed {len(runs)} runs. Compact outputs saved to: {out}")
    print("Main report:",out/"comparison.md")


if __name__=="__main__":
    main()
