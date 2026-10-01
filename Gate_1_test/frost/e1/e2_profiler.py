"""E2 cost decomposition profiler for the current MACE/Frost path.

This is a bounded profiler, not an MD run and not a speedup claim. It uses
CUDA events where available, saved Stage-A frames, and never materializes a
dense layer-1 Jacobian.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from frost.e1.force_error import (
    layer1_edge_jvp,
    layer1_edge_vjp,
    extract_layer1_local_inputs,
)
from frost.e1.geometry import build_edge_geometry_from_ase, is_dirty_zeroth_order
from frost.e1.mace_patch import (
    apply_frost_patch,
    create_patched_forward,
    substitute_cached_messages,
)
from frost.e1.model_interface import load_model
from frost.e1.reference_cache import PerEdgeReferenceCache


def parse_steps(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "raw_ms": [float(value) for value in array],
        "median_ms": float(np.median(array)),
        "iqr_ms": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10_ms": float(np.percentile(array, 10)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def cuda_or_wall_time(function):
    wall_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0
    result = function()
    elapsed = (time.perf_counter() - wall_start) * 1000.0
    return result, elapsed, elapsed


def model_batch(interface, atoms, requires_grad: bool):
    calculator = interface.calculator
    batch = calculator._clone_batch(calculator._atoms_to_batch(atoms))
    dtype = next(calculator.models[0].parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=dtype)
    if requires_grad:
        batch["positions"].requires_grad_(True)
    return batch


def profiled_forward(interface, atoms, compute_force=False, requires_grad=None):
    model = interface.calculator.models[0]
    if requires_grad is None:
        requires_grad = compute_force
    batch = model_batch(interface, atoms, requires_grad=requires_grad)
    profile = {}
    state = {
        "active": True,
        "mode": 0,
        "clean_mask": torch.empty(0, dtype=torch.bool),
        "g_ref": torch.empty(0),
        "profile": profile,
    }

    def forward():
        with apply_frost_patch(model, state):
            output = model(batch.to_dict(), compute_force=compute_force)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return output

    output, elapsed, cpu_elapsed = cuda_or_wall_time(forward)
    profile["total_forward_ms"] = elapsed
    profile["total_forward_cpu_wall_ms"] = cpu_elapsed
    return output, batch, profile


def backward_force(batch, output):
    energy = output["energy"].sum()

    def backward():
        return torch.autograd.grad(energy, batch["positions"], retain_graph=False)[0]

    _, elapsed, _ = cuda_or_wall_time(backward)
    return elapsed


def finish_profile_events(profile):
    result = {}
    for name, events in profile.items():
        if name.endswith("_cpu_ms"):
            result[name] = [float(value) for value in events]
            continue
        if not name.endswith("_events"):
            continue
        key = name.removesuffix("_events")
        result[key + "_ms"] = [float(start.elapsed_time(end)) for start, end in events]
    if "total_forward_ms" in profile:
        result["total_forward_ms"] = profile["total_forward_ms"]
    if "total_forward_cpu_wall_ms" in profile:
        result["total_forward_cpu_wall_ms"] = profile["total_forward_cpu_wall_ms"]
    return result


def profile_component(profile, name):
    if torch.cuda.is_available():
        return profile.get(name + "_ms", [])
    return profile.get(name + "_cpu_ms", [])


def time_microbenchmarks(interface, atoms, local, repeats, warmup):
    model = interface.calculator.models[0]
    edge_count = local["vectors"].shape[0]
    feature_count = local["mji"].shape[1]
    sample_count = min(16, edge_count)
    directions = torch.linspace(
        -0.001, 0.001, sample_count * 3,
        device=local["vectors"].device,
        dtype=local["vectors"].dtype,
    ).reshape(sample_count, 3)
    upstream = torch.linspace(
        -0.2, 0.2, sample_count * feature_count,
        device=local["mji"].device,
        dtype=local["mji"].dtype,
    ).reshape(sample_count, feature_count)

    def jvp_batch():
        return [layer1_edge_jvp(interface, local, edge, directions[edge]) for edge in range(sample_count)]

    def vjp_batch():
        return [layer1_edge_vjp(interface, local, edge, upstream[edge]) for edge in range(sample_count)]

    clean_mask = torch.zeros(edge_count, dtype=torch.bool, device=local["mji"].device)
    clean_mask[: max(1, edge_count // 2)] = True
    dirty_indices = torch.nonzero(~clean_mask, as_tuple=False).flatten()
    dirty_messages = local["mji"][dirty_indices]
    reference_messages = local["mji"].clone()
    correction = torch.zeros_like(reference_messages)
    current_vectors = local["vectors"]
    reference_vectors = current_vectors - 0.001
    geometry = build_edge_geometry_from_ase(atoms, interface.cutoff)
    reference_entries = {
        key: {
            "ref_r_ij": value["r_ij"],
            "ref_d_ij": value["d_ij"],
            "ref_rhat_ij": value["rhat_ij"],
        }
        for key, value in geometry.items()
    }
    perturbations = {key: {"delta_r": 0.01, "delta_rhat": 0.0, "new": False} for key in geometry}
    keys = list(geometry)

    def affine_update():
        return reference_messages + correction

    def predicate():
        return [
            is_dirty_zeroth_order(
                geometry[key]["d_ij"],
                geometry[key]["rhat_ij"],
                reference_entries[key]["ref_d_ij"],
                reference_entries[key]["ref_rhat_ij"],
                0.01,
            )
            for key in keys
        ]

    def compaction():
        return torch.nonzero(~clean_mask, as_tuple=False).flatten()

    def reconstruction():
        output = reference_messages.clone()
        output.index_copy_(0, dirty_indices, dirty_messages)
        return output

    edge_index = local["edge_index"][:, dirty_indices]
    edge_attrs = local["edge_attrs"][dirty_indices]
    edge_feats = local["edge_feats"][dirty_indices]
    cutoff = local["cutoff"][dirty_indices] if local["cutoff"] is not None else None
    source = local["node_feats"][edge_index[0]]

    def dirty_tp():
        weights = model.interactions[0].conv_tp_weights(edge_feats)
        if cutoff is not None:
            weights = weights * cutoff
        return model.interactions[0].conv_tp(source, edge_attrs, weights)

    functions = {
        "JVP_16_edges": jvp_batch,
        "VJP_16_edges": vjp_batch,
        "affine_cache_update": affine_update,
        "predicate": predicate,
        "compaction": compaction,
        "cache_reconstruction": reconstruction,
        "dirty_TP": dirty_tp,
    }
    measurements = {}
    for name, function in functions.items():
        for _ in range(warmup):
            function()
        gpu_samples = []
        cpu_samples = []
        for _ in range(repeats):
            _, gpu_ms, cpu_ms = cuda_or_wall_time(function)
            gpu_samples.append(gpu_ms)
            cpu_samples.append(cpu_ms)
        measurements[name] = {
            "cuda": statistics(gpu_samples),
            "cpu_wall": statistics(cpu_samples),
        }
    return measurements


def profile_frame(interface, atoms, repeats, warmup):
    for _ in range(warmup):
        profiled_forward(interface, atoms, compute_force=False)
        output, batch, _ = profiled_forward(interface, atoms, compute_force=False, requires_grad=True)
        backward_force(batch, output)

    measurements = {name: [] for name in (
        "T_exact_total", "T_exact_total_cpu_wall", "T_layer1_TP",
        "T_layer1_message", "T_layer1_aggregation", "T_forward_cpu_wall",
        "T_rest_forward", "T_rest_forward_cpu_wall", "T_backward_force",
        "T_edge_geometry", "T_total_step_wall",
    )}
    for _ in range(repeats):
        wall_start = time.perf_counter()
        def exact_evaluation():
            interface.calculator.results = {}
            return interface.evaluate(atoms)

        _, exact_total, exact_cpu_wall = cuda_or_wall_time(exact_evaluation)
        _, forward, profile = profiled_forward(interface, atoms, compute_force=False)
        forward_profile = finish_profile_events(profile)
        output, batch, force_profile = profiled_forward(interface, atoms, compute_force=False, requires_grad=True)
        backward_ms = backward_force(batch, output)
        tp = sum(profile_component(forward_profile, "layer1_tp"))
        message = sum(profile_component(forward_profile, "layer1_message"))
        aggregation = sum(profile_component(forward_profile, "layer1_aggregation"))
        forward_ms = forward_profile["total_forward_ms"]
        forward_cpu_wall = forward_profile["total_forward_cpu_wall_ms"]
        measurements["T_exact_total"].append(exact_total)
        measurements["T_exact_total_cpu_wall"].append(exact_cpu_wall)
        measurements["T_layer1_TP"].append(tp)
        measurements["T_layer1_message"].append(message)
        measurements["T_layer1_aggregation"].append(aggregation)
        measurements["T_forward_cpu_wall"].append(forward_cpu_wall)
        measurements["T_rest_forward"].append(max(forward_ms - message - aggregation, 0.0))
        measurements["T_rest_forward_cpu_wall"].append(max(forward_cpu_wall - message - aggregation, 0.0))
        measurements["T_backward_force"].append(backward_ms)
        edge_start = time.perf_counter()
        build_edge_geometry_from_ase(atoms, interface.cutoff)
        measurements["T_edge_geometry"].append((time.perf_counter() - edge_start) * 1000.0)
        measurements["T_total_step_wall"].append((time.perf_counter() - wall_start) * 1000.0)
    return {name: statistics(values) for name, values in measurements.items()}


def write_summary(path, metadata, frame_results, microbenchmarks):
    theta_inputs = [
        frame["measurements"]["T_layer1_message"]["median_ms"]
        for frame in frame_results
    ]
    aggregation_inputs = [
        frame["measurements"]["T_layer1_aggregation"]["median_ms"]
        for frame in frame_results
    ]
    tp = float(np.median(theta_inputs))
    aggregation = float(np.median(aggregation_inputs))
    denominator = tp + aggregation
    metadata["theta_numerator_ms"] = tp
    metadata["theta_denominator_ms"] = denominator
    metadata["theta"] = tp / max(denominator, 1e-30)
    metadata["theta_definition"] = "median complete layer-1 message construction / (median complete message construction + median scatter aggregation)"
    summary = {
        "metadata": metadata,
        "call_graph": metadata["call_graph"],
        "frames": frame_results,
        "microbenchmarks": microbenchmarks,
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    with (path / "frame_measurements.csv").open("w", newline="") as handle:
        fields = ["step", "n_edges"] + list(frame_results[0]["measurements"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frame_results:
            row = {"step": frame["step"], "n_edges": frame["n_edges"]}
            row.update({name: values["median_ms"] for name, values in frame["measurements"].items()})
            writer.writerow(row)
    lines = [
        "# Frost E2 profiler summary",
        "",
        "This report measures execution components and action/bookkeeping costs. It is not an end-to-end speedup result.",
        "",
        f"- theta: {metadata['theta']:.6f}",
        f"- theta numerator (message median, ms): {metadata['theta_numerator_ms']:.9f}",
        f"- theta denominator (message + aggregation medians, ms): {metadata['theta_denominator_ms']:.9f}",
        f"- device: {metadata['device']}",
        f"- dtype: {metadata['dtype']}",
        f"- frames: {metadata['steps']}",
        "",
        "Dense Jacobian extraction was not used.",
        "",
        "## Call graph and source boundaries",
    ]
    for name, location in metadata["call_graph"].items():
        lines.append(f"- {name}: `{location}`")
    lines.extend(["", "## Timer accounting"])
    for name, scope in metadata["timer_accounting"].items():
        lines.append(f"- **{name}**: {scope}")
    (path / "summary.md").write_text("\n".join(lines) + "\n")


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    interface = load_model("mace-mp-0", device=args.device, dtype=args.dtype)
    frames = read(str(args.trajectory), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    steps = parse_steps(args.steps)
    selected = [(step, frames[step]) for step in steps]
    frame_results = []
    all_micro = []
    for step, atoms in selected:
        local = extract_layer1_local_inputs(interface, atoms)
        if not frame_results:
            parameter = next(interface.calculator.models[0].parameters())
            positions = model_batch(interface, atoms, requires_grad=False)["positions"]
            print(f"[e2] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
            print(f"[e2] requested_device={args.device} selected_device={interface.device}", flush=True)
            print(f"[e2] model_parameter_device={parameter.device}", flush=True)
            print(f"[e2] positions_device={positions.device} dtype={positions.dtype}", flush=True)
            print(f"[e2] timing_clock={'cuda_events' if torch.cuda.is_available() else 'cpu_perf_counter'}", flush=True)
        measurements = profile_frame(interface, atoms, args.repeats, args.warmup)
        micro = time_microbenchmarks(interface, atoms, local, args.repeats, args.warmup)
        frame_results.append({"step": step, "n_edges": int(local["vectors"].shape[0]), "measurements": measurements})
        all_micro.append({"step": step, "measurements": micro})
        print(f"[e2] step={step} edges={local['vectors'].shape[0]} measured", flush=True)
        for name in (
            "T_exact_total",
            "T_layer1_message",
            "T_layer1_aggregation",
            "T_rest_forward",
            "T_backward_force",
            "T_edge_geometry",
        ):
            stats = measurements[name]
            print(
                f"[e2] step={step} {name}: n={stats['n']} "
                f"raw_ms={stats['raw_ms']} median_ms={stats['median_ms']:.6f} "
                f"iqr_ms={stats['iqr_ms']:.6f} p10_ms={stats['p10_ms']:.6f} "
                f"p90_ms={stats['p90_ms']:.6f}",
                flush=True,
            )
    metadata = {
        "stage": "E2",
        "device": interface.device,
        "dtype": args.dtype,
        "steps": steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dense_jacobian_enabled": False,
        "stage_b_runtime_used": False,
        "n_edges_expected": 2944,
        "message_features_expected": 2048,
        "timing_clock": "cuda_events" if torch.cuda.is_available() else "cpu_perf_counter",
        "timer_accounting": {
            "T_exact_total": "CUDA elapsed around calculator.evaluate when CUDA is available, otherwise CPU perf_counter; calculator results are cleared; separate CPU wall counterpart is T_exact_total_cpu_wall",
            "T_layer1_TP": "CUDA event or CPU perf_counter around first-layer conv_tp only; nested inside T_layer1_message; excludes conv_tp_weights and cutoff",
            "T_layer1_message": "CUDA event or CPU perf_counter from before first-layer conv_tp_weights through mji construction; includes radial tensor-product weights, cutoff multiplication, and conv_tp; nested inside forward",
            "T_layer1_aggregation": "CUDA event or CPU perf_counter around first-layer scatter_sum only; nested inside forward and disjoint from T_layer1_message",
            "T_rest_forward": "timed model-forward elapsed on the active clock minus complete layer-1 message and scatter events; excludes model_batch() batch/neighbor preparation, and includes layers 2+, readout, and remaining layer-1 work",
            "T_backward_force": "CUDA elapsed or CPU perf_counter for explicit autograd.grad of energy with respect to positions; separate from forward",
            "T_edge_geometry": "CPU perf_counter around Frost build_edge_geometry_from_ase only; includes ASE neighbor-list construction and MIC geometry",
            "T_total_step_wall": "CPU perf_counter enclosing exact evaluation, profiled forward, profiled force forward/backward, and edge geometry; includes host overhead and synchronization",
            "microbenchmarks": "each operation has separate CUDA elapsed and CPU wall distributions; JVP/VJP are serial 16-edge action samples; dirty_TP is a 50-percent edge subset",
        },
        "call_graph": {
            "ASE/MACE batch and neighbor preparation": f"{inspect.getsourcefile(interface.calculator._atoms_to_batch)}:{inspect.getsourcelines(interface.calculator._atoms_to_batch)[1]}",
            "Frost edge geometry": f"{inspect.getsourcefile(build_edge_geometry_from_ase)}:{inspect.getsourcelines(build_edge_geometry_from_ase)[1]}",
            "MACE calculator evaluation": f"{inspect.getsourcefile(interface.calculator.calculate)}:{inspect.getsourcelines(interface.calculator.calculate)[1]}",
            "MACE model forward/readout": f"{inspect.getsourcefile(interface.calculator.models[0].forward)}:{inspect.getsourcelines(interface.calculator.models[0].forward)[1]}",
            "MACE interaction block forward": f"{inspect.getsourcefile(interface.calculator.models[0].interactions[0].forward)}:{inspect.getsourcelines(interface.calculator.models[0].interactions[0].forward)[1]}",
            "MACE subsequent interaction layers": f"{inspect.getsourcefile(interface.calculator.models[0].interactions[0].forward)}:same InteractionBlock.forward for interactions[1:]",
            "MACE first interaction/Frost interception": f"{inspect.getsourcefile(create_patched_forward)}:{inspect.getsourcelines(create_patched_forward)[1]}",
            "layer-1 tensor product and scatter timing": f"{inspect.getsourcefile(create_patched_forward)}:{inspect.getsourcelines(create_patched_forward)[1]}",
            "Frost affine message substitution": f"{inspect.getsourcefile(substitute_cached_messages)}:{inspect.getsourcelines(substitute_cached_messages)[1]}",
            "Frost JVP action": f"{inspect.getsourcefile(layer1_edge_jvp)}:{inspect.getsourcelines(layer1_edge_jvp)[1]}",
            "Frost VJP action": f"{inspect.getsourcefile(layer1_edge_vjp)}:{inspect.getsourcelines(layer1_edge_vjp)[1]}",
            "dirty predicate": f"{inspect.getsourcefile(is_dirty_zeroth_order)}:{inspect.getsourcelines(is_dirty_zeroth_order)[1]}",
            "cache state machine": f"{inspect.getsourcefile(PerEdgeReferenceCache)}:{inspect.getsourcelines(PerEdgeReferenceCache)[1]}",
        },
    }
    write_summary(output, metadata, frame_results, all_micro)
    print(f"[e2] theta numerator_ms={metadata['theta_numerator_ms']:.9f}", flush=True)
    print(f"[e2] theta denominator_ms={metadata['theta_denominator_ms']:.9f}", flush=True)
    print(f"[e2] theta={metadata['theta']:.9f}", flush=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Frost E2 MACE cost profiler")
    parser.add_argument("--trajectory", default="results/e1/temporal_redundancy_100eq_1000prod/trajectory.traj")
    parser.add_argument("--steps", default="0,40")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", default="results/e2/profiler")
    run(parser.parse_args())
