# Frost end-to-end benchmark

This short benchmark compares exact MACE and the integrated mode-3 Frost calculator. Timed steps include the complete ASE MD step and all Frost cache/action work. Dense Jacobians and Stage-B replay are disabled.

Command: `python -m frost.e1.end_to_end_benchmark --output-dir results/e2/end_to_end_frost_short`
