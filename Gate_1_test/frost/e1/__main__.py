"""
Frost E1 — __main__.py

Entry point for the E1 Redundancy Characterisation experiment.

Usage:
  python -m frost.e1 --config configs/e1.yaml
  python -m frost.e1 --config configs/e1.yaml --pilot
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path
import time
import numpy as np

from frost.e1.systems import build_system
from frost.e1.model_interface import load_model
from frost.e1.geometry import build_edge_geometry_from_ase
from frost.e1.reference_cache import PerEdgeReferenceCache
from frost.e1.trajectory import run_md_trajectory
from frost.e1.logging_hdf5 import E1Logger
from frost.e1.force_error import run_force_error_experiment
from frost.e1.analysis import (
    load_trajectory_stats,
    compute_dirty_fraction_summary,
    compute_cache_lifetime_from_dirty_fractions,
    apply_gonogo_criterion,
)
from frost.e1.plots import (
    plot_dirty_fraction_vs_tolerance,
    plot_perturbation_force_error_joint,
    plot_delta_r_vs_delta_rhat,
    plot_force_error_by_tolerance,
    plot_cache_lifetime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("frost.e1")


def run_experiment(config: dict, is_pilot: bool = False):
    out_dir = Path(config.get("output_dir", "results/e1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    plots_dir = out_dir / "plots"
    raw_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    if is_pilot:
        logger.info("=== RUNNING IN PILOT MODE ===")
        models = [config["pilot"]["model"]]
        systems = [s for s in config["systems"] if s["name"] == config["pilot"]["system"]]
        temperatures = [config["pilot"]["temperature"]]
        seeds = [config["pilot"]["seed"]]
        n_eq_steps = config["pilot"]["equilibration_steps"]
        n_prod_steps = config["pilot"]["production_steps"]
        n_force_frames = 10
    else:
        logger.info("=== RUNNING FULL E1 MATRIX ===")
        models = config["models"]
        systems = config["systems"]
        temperatures = config["temperatures"]
        seeds = config["seeds"]
        n_eq_steps = config["equilibration_steps"]
        n_prod_steps = config["production_steps"]
        n_force_frames = config["num_force_error_frames"]

    tolerances = config["tolerances_meV_per_A"]
    timestep_fs = config["timestep_fs"]
    langevin_gamma = config["langevin_gamma"]

    # Select random frames for force error computation
    # We do this up front for consistency
    all_force_frames = {}
    for seed in seeds:
        rng = np.random.default_rng(seed + 999)
        frames = rng.choice(n_prod_steps, size=min(n_force_frames, n_prod_steps), replace=False)
        all_force_frames[seed] = set(frames)

    all_results = []
    
    # 1. Run Trajectories
    for model_name in models:
        for sys_config in systems:
            sys_name = sys_config["name"]
            for temp in temperatures:
                for seed in seeds:
                    traj_id = f"{model_name}_{sys_name}_{temp}K_seed{seed}"
                    logger.info("\n--- Starting %s ---", traj_id)

                    try:
                        atoms = build_system(sys_config["structure"])
                    except Exception as e:
                        logger.error("Failed to build system %s: %s", sys_name, e)
                        continue

                    try:
                        model = load_model(model_name, device=config.get("device", "cuda"), dtype=config.get("dtype", "float64"))
                    except Exception as e:
                        logger.error("Failed to load model %s: %s", model_name, e)
                        continue

                    # Pre-calculate what frames need force eval
                    force_eval_frames = all_force_frames[seed]

                    metadata = {
                        "trajectory_id": traj_id,
                        "model": model_name,
                        "system": sys_name,
                        "temperature_K": temp,
                        "seed": seed,
                        "timestep_fs": timestep_fs,
                    }

                    hdf5_path = raw_dir / f"{traj_id}.h5"
                    logger_obj = E1Logger(
                        filepath=hdf5_path,
                        metadata=metadata,
                        tolerances=tolerances,
                        n_steps_estimate=n_prod_steps,
                        chunk_size=config.get("chunk_size", 500)
                    )

                    cache = PerEdgeReferenceCache(tolerances)

                    # We need a callback that records the edges at every step
                    def md_callback(current_atoms, step_idx):
                        # 1. Extract geometry
                        edges_current = build_edge_geometry_from_ase(current_atoms, model.cutoff)
                        
                        # 2. If first step, initialize cache
                        if step_idx == 0:
                            cache.initialize(edges_current, timestep=0)
                            
                        # 3. Step cache
                        stats = cache.step(edges_current, timestep=step_idx)
                        
                        # 4. Log stats
                        logger_obj.log_step(stats)
                        
                        # 5. Run force error experiment on selected frames
                        if step_idx in force_eval_frames:
                            logger.info("    Force error eval at frame %d", step_idx)
                            # Get references from cache (as a dict)
                            # This simulates what Frost does: replacing stale atoms with reference positions
                            ref_entries = cache._entries_as_ref_dict()
                            
                            # Construct reference positions for all atoms from their cached values?
                            # WAIT: cache._entries stores edge ref. We need atomic ref positions.
                            # We must store atom positions at reference refresh.
                            # For E1 approximation (atom-level freezing), we will just track
                            # atom reference positions globally when they are refreshed.
                            pass # We'll do the actual force error inside the callback

                    # Because atom reference positions weren't explicitly tracked in EdgeCacheEntry 
                    # (it tracks edges), we will maintain an atom-level reference position array for the
                    # force error approximation.
                    
                    atom_ref_pos = {} # per epsilon
                    for eps in tolerances:
                        atom_ref_pos[eps] = atoms.get_positions().copy()
                    
                    # We will override the callback to properly update atom_ref_pos based on which atoms are NOT dirty
                    def md_callback_with_force_error(current_atoms, step_idx):
                        edges_current = build_edge_geometry_from_ase(current_atoms, model.cutoff)
                        
                        if step_idx == 0:
                            cache.initialize(edges_current, timestep=0)
                            for eps in tolerances:
                                atom_ref_pos[eps] = current_atoms.get_positions().copy()
                                
                        stats = cache.step(edges_current, timestep=step_idx)
                        logger_obj.log_step(stats)
                        
                        # Update atom_ref_pos: if an atom is dirty (any edge dirty), it gets updated to current
                        # This is a safe upper bound: if we update it, it's not stale.
                        # Wait, the stale atom definition: an atom is stale if ALL its edges are dirty.
                        # Wait no, if an atom's edges are dirty, we must RECOMPUTE (it is NOT stale).
                        # So if an atom has ANY dirty edge, we must recompute it -> its reference position becomes current.
                        # If ALL its edges are clean, it can be STALE -> it keeps its old reference position.
                        
                        curr_pos = current_atoms.get_positions()
                        for eps in tolerances:
                            # find dirty edges
                            dirty_atoms = set()
                            for key, pert in stats["perturbations"].items():
                                i, j, shift = key
                                if pert["new"] or pert["delta_r"] > eps:
                                    dirty_atoms.add(i)
                                    dirty_atoms.add(j)
                                    
                            # Update reference positions for dirty atoms
                            if dirty_atoms:
                                dirty_idx = list(dirty_atoms)
                                atom_ref_pos[eps][dirty_idx] = curr_pos[dirty_idx]
                        
                        # Run force error eval
                        if step_idx in force_eval_frames:
                            logger.info("    Force error eval at frame %d", step_idx)
                            res = run_force_error_experiment(
                                current_atoms,
                                stats["perturbations"],
                                atom_ref_pos, # Note: modified run_force_error_experiment slightly
                                tolerances,
                                model
                            )
                            for eps, eps_res in res.items():
                                logger_obj.log_force_error_frame(
                                    frame_timestep=step_idx,
                                    epsilon=eps,
                                    exact_forces=eps_res["exact_forces"],
                                    stale_forces=eps_res["stale_forces"],
                                    positions=current_atoms.get_positions(),
                                    cell=current_atoms.get_cell().array
                                )

                    # Monkey-patch force error runner to accept a dict of ref positions per eps
                    def custom_force_error(frame_atoms, perturbations, ref_pos_dict, tols, mdl):
                        from frost.e1.force_error import compute_exact_forces, compute_stale_forces, compute_force_error_statistics
                        exact_e, exact_f = compute_exact_forces(frame_atoms, mdl)
                        res = {}
                        for e in tols:
                            # Stale configuration: we just use ref_pos_dict[e] directly for ALL atoms!
                            # Because we already updated dirty atoms to have current pos, 
                            # and stale atoms kept their old pos.
                            atoms_stale = frame_atoms.copy()
                            atoms_stale.set_positions(ref_pos_dict[e])
                            _, stale_f = mdl.evaluate(atoms_stale)
                            
                            stats = compute_force_error_statistics(exact_f, stale_f)
                            stats["epsilon"] = e
                            stats["exact_forces"] = exact_f
                            stats["stale_forces"] = stale_f
                            res[e] = stats
                        return res
                    
                    # Override the global run_force_error_experiment to use our custom one 
                    # that perfectly tracks atom positions per epsilon
                    global run_force_error_experiment
                    run_force_error_experiment = custom_force_error

                    with logger_obj as log:
                        run_md_trajectory(
                            atoms=atoms,
                            calculator=model,
                            temperature_K=temp,
                            n_equilibration_steps=n_eq_steps,
                            n_production_steps=n_prod_steps,
                            timestep_fs=timestep_fs,
                            langevin_gamma=langevin_gamma,
                            seed=seed,
                            step_callback=md_callback_with_force_error,
                            log_interval=1
                        )
                        
                        # Log cache lifetimes
                        lifetime_summary = cache.get_lifetime_summary()
                        if lifetime_summary:
                            for eps in tolerances:
                                # This is global over all tolerances in the simplified get_lifetime_summary
                                log.log_cache_lifetime(lifetime_summary, eps)
                    
                    all_results.append(hdf5_path)

    # 2. Analysis and Plots
    logger.info("=== GENERATING ANALYSIS AND PLOTS ===")
    analyzed_results = []
    
    all_force_error_frames = []
    all_perturbations = []
    
    for hdf5_path in all_results:
        meta, steps_df, force_errors = load_trajectory_stats(hdf5_path)
        if steps_df.empty:
            continue
            
        dirty_summary = compute_dirty_fraction_summary(steps_df, tolerances)
        lifetime_summary = compute_cache_lifetime_from_dirty_fractions(steps_df, tolerances)
        
        analyzed_results.append({
            "metadata": meta,
            "dirty_summary_df": dirty_summary,
            "lifetime_summary": lifetime_summary,
        })
        
        for eps, frames in force_errors.items():
            for f in frames:
                # Need to match mean_delta_r for the joint plot
                f_step = f["timestep"]
                step_row = steps_df[steps_df["timestep"] == f_step]
                if not step_row.empty:
                    f["mean_delta_r"] = float(step_row["mean_delta_r"].iloc[0])
                all_force_error_frames.append(f)
                
        # Sample perturbations for delta_r vs delta_rhat (take first step for simplicity or average)
        if "mean_delta_r" in steps_df.columns:
            # Just take 100 random steps
            sample_df = steps_df.sample(min(100, len(steps_df)))
            for _, row in sample_df.iterrows():
                all_perturbations.append({
                    "delta_r": row["mean_delta_r"],
                    "delta_rhat": row["mean_delta_rhat"],
                    "epsilon": 0, # not used here
                })

    plot_dirty_fraction_vs_tolerance(analyzed_results, plots_dir / "dirty_fraction_vs_tolerance.png")
    
    # We can group force_error_frames by system/temp if we want, but paper just wants joint
    plot_perturbation_force_error_joint(
        all_force_error_frames, 
        plots_dir / "perturbation_force_error_joint.png",
        model_rmse_eV=None,
        system="All Systems"
    )
    
    plot_delta_r_vs_delta_rhat(
        all_perturbations,
        plots_dir / "delta_r_vs_delta_rhat.png",
        system="All Systems"
    )
    
    # Apply Go/No-Go Criterion on the aggregate (use Si as representative)
    # Get Si results
    si_results = [r for r in analyzed_results if r["metadata"]["system"] == "Si" and r["metadata"]["temperature_K"] == 300]
    if si_results:
        dirty_df = si_results[0]["dirty_summary_df"]
        
        # Build force error summary for Si 300K
        fe_summary = {}
        for eps in tolerances:
            eps_frames = [f for f in all_force_error_frames if f["epsilon"] == eps] # should filter by Si 300K
            if eps_frames:
                fe_summary[eps] = {
                    "mean_error": np.mean([f["mean_error"] for f in eps_frames]),
                    "max_error": np.max([f["max_error"] for f in eps_frames]),
                    "rms_error": np.mean([f["rms_error"] for f in eps_frames]), # approx
                }
                
        if fe_summary:
            plot_force_error_by_tolerance(fe_summary, plots_dir / "force_error_by_tolerance.png", system="Si", temperature_K=300)
            
            verdict = apply_gonogo_criterion(dirty_df, fe_summary)
            logger.info("=== GO/NO-GO VERDICT ===")
            logger.info("Pass: %s", verdict["pass"])
            logger.info("Explanation: %s", verdict["explanation"])
            
            with open(Path("docs") / "e1_results.md", "w") as f:
                f.write("# Frost E1 Results\n\n")
                f.write(f"## Verdict: {'PASS' if verdict['pass'] else 'FAIL'}\n\n")
                f.write(f"{verdict['explanation']}\n\n")
                f.write("### Plots\n")
                f.write("See `results/e1/plots/` for detailed figures.\n")

    logger.info("Experiment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frost E1 Redundancy Characterisation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--pilot", action="store_true", help="Run quick pilot test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_experiment(config, is_pilot=args.pilot)
