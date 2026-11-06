# Optimization log

| Change | Validation | Status |
|---|---|---|
| Chunked batched first-order JVP/VJP path (mode 4) | Existing unit action tests; GPU mode-3/mode-4 comparison still required | Available for validation |
| Canonical edge-key matching | Geometry and edge identity tests; mismatches raise | Retained |
| Synchronized complete-evaluation W4 timing | Protocol implemented in `inference_benchmark.py`; no GPU run performed here | Available for validation |
| Aggregate-only edge analysis guard | Unit-testable schema validation; no fabricated per-edge values | Retained |

No unmeasured runtime improvement is claimed.
