import csv

import pytest

from frost.e1.cost_model_analysis import analyze as analyze_costs
from frost.e1.cost_model_analysis import predict_runtime
from frost.e1.edge_error_analysis import analyze as analyze_edges


def test_cost_model_rejects_negative_components():
    with pytest.raises(ValueError):
        predict_runtime(1.0, -0.1, 0.0, 0.0, 0.0, 0.0)


def test_cost_model_writes_measured_residual(tmp_path):
    source = tmp_path / "cost.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "arm", "baseline_seconds", "targeted_seconds_saved", "dirty_seconds",
            "derivative_seconds", "geometry_seconds", "cache_seconds",
            "synchronization_seconds", "measured_seconds",
        ])
        writer.writeheader()
        writer.writerow({
            "arm": "first_order", "baseline_seconds": 10, "targeted_seconds_saved": 4,
            "dirty_seconds": 1, "derivative_seconds": 2, "geometry_seconds": 1,
            "cache_seconds": 1, "synchronization_seconds": 0.5, "measured_seconds": 12,
        })
    rows = analyze_costs(source, tmp_path / "out.csv")
    assert rows[0]["predicted_seconds"] == pytest.approx(11.5)
    assert rows[0]["prediction_residual_seconds"] == pytest.approx(0.5)


def test_edge_analysis_rejects_aggregate_stage_b_rows(tmp_path):
    source = tmp_path / "aggregate.csv"
    source.write_text("step,tolerance_A,edge_message_clean_rms\n0,0.01,0.2\n")
    with pytest.raises(ValueError, match="aggregate result"):
        analyze_edges(source, tmp_path / "edge")
