"""Fixed-configuration full-calculator benchmark (W4)."""
from __future__ import annotations
import argparse, csv, json, time, subprocess
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from frost.e1.end_to_end_benchmark import IntegratedFrostCalculator
from frost.e1.model_interface import load_model


def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize()

def run(args):
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_frames=read(args.trajectory, index=":")
    chosen_steps=args.steps[:args.n_frames]
    if not chosen_steps or min(chosen_steps)<0 or max(chosen_steps)>=len(all_frames):
        raise ValueError("Requested frame step is outside the trajectory")
    frames=[all_frames[step] for step in chosen_steps]
    model=load_model(args.model, device=args.device, dtype=args.dtype)
    arms=[("exact",None,"disabled",0.0), ("frost_disabled",1,"disabled",0.0),
          ("frost_dirty_only",1,"dirty_only",0.0),
          *((f"frost_reuse_{int(f*100)}pct",1,"fraction",f) for f in (0.0,.25,.5,.75,1.0)),
          ("frost_first_order",3,"threshold",1.0)]
    rows=[]; component_rows=[]; memory_peaks={}
    for frame_index, base in enumerate(frames):
        for name, mode, policy, reuse_fraction in arms:
            for repeat in range(args.repeats):
                atoms=base.copy()
                if name=="exact": atoms.calc=model.calculator
                else: atoms.calc=IntegratedFrostCalculator(model,args.tolerance_A,cache_mode=mode,cache_policy=policy,reuse_fraction=reuse_fraction)
                for _ in range(args.warmup):
                    atoms.calc.results={}; atoms.get_forces(); sync(model.device)
                if name != "exact":
                    atoms.calc.begin_timing()
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                for iteration in range(args.evaluations):
                    atoms.calc.results={}
                    sync(model.device); start=time.perf_counter()
                    atoms.get_forces(); sync(model.device)
                    elapsed=time.perf_counter()-start
                    rows.append({"frame":chosen_steps[frame_index],"arm":name,"repeat":repeat,"iteration":iteration,"wall_seconds":elapsed})
                if name != "exact":
                    component_rows.extend({"frame":chosen_steps[frame_index],"arm":name,"repeat":repeat,**record} for record in atoms.calc.records)
                if torch.cuda.is_available():
                    key=(name,repeat,chosen_steps[frame_index])
                    memory_peaks[str(key)]=int(torch.cuda.max_memory_allocated())
    with (out/"timings.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    exact_median=np.median([r["wall_seconds"] for r in rows if r["arm"]=="exact"])
    for name,_,_,_ in arms:
        x=np.asarray([r["wall_seconds"] for r in rows if r["arm"]==name])
        phase={}
        for field in ("geometry_seconds","message_input_seconds","cache_update_seconds","model_evaluation_seconds"):
            values=[r[field] for r in component_rows if r["arm"]==name]
            if values: phase[field+"_mean"]=float(np.mean(values))
        summary.append({"arm":name,"n":len(x),"median_seconds":float(np.median(x)),"speedup_vs_exact_median":float(exact_median/np.median(x)),"iqr_seconds":float(np.percentile(x,75)-np.percentile(x,25)),
                        "p90_seconds":float(np.percentile(x,90)),"p95_seconds":float(np.percentile(x,95)),"throughput_evals_s":float(1/x.mean()),**phase})
    payload={"experiment":"fixed_configuration_calculator_benchmark","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "model":args.model,"device":model.device,"dtype":args.dtype,"frames":chosen_steps,"warmup":args.warmup,"evaluations_per_frame_arm":args.evaluations,
      "repeats":args.repeats,"synchronization":"CUDA synchronized immediately before and after each full calculator evaluation","summaries":summary,
      "peak_cuda_allocated_bytes_by_arm_repeat_frame":memory_peaks,
      "gpu_utilization":"not sampled; timing is synchronized wall time",
      "limitations":"Each arm uses fixed saved configurations; this does not measure MD rollout cost. Component times are host wall slices; use the synchronized full-calculator wall time as the performance result."}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
    with (out/"component_timings.csv").open("w",newline="") as f:
        if component_rows:
            w=csv.DictWriter(f,fieldnames=list(component_rows[0])); w.writeheader(); w.writerows(component_rows)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trajectory",required=True); p.add_argument("--steps",type=int,nargs="+",default=[0]);
    p.add_argument("--n-frames",type=int,default=5); p.add_argument("--warmup",type=int,default=10); p.add_argument("--evaluations",type=int,default=100)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--tolerance-A",type=float,default=.01); p.add_argument("--model",default="mace-mp-0"); p.add_argument("--device",default="cuda")
    p.add_argument("--dtype",choices=["float32","float64"],default="float64"); p.add_argument("--output-dir",default="results/active/w4_inference")
    run(p.parse_args())
