# Performance methodology

The primary performance quantity is synchronized complete-calculator wall
time. Component slices explain overhead but must not be summed when they
overlap. W4 performs warmups, synchronizes before and after each timed GPU
evaluation, records repeated observations, and reports median, IQR, P90, and
P95.

Report exact MACE, disabled Frost, dirty-only, controlled clean fractions,
natural threshold reuse, and first-order reuse separately. Controlled fractions
are diagnostic controls, not policies.

Peak CUDA allocated memory is recorded where CUDA is available. The cost model
in [`cost_model_analysis.py`](../frost/e1/cost_model_analysis.py) distinguishes
theoretical, predicted, and measured speedups; it never invents a break-even
point from missing measurements.
