# Frost E2 profiler summary

This report measures execution components and action/bookkeeping costs. It is not an end-to-end speedup result.

- theta: 0.830599
- theta numerator (message median, ms): 8.918233507
- theta denominator (message + aggregation medians, ms): 10.737108241
- device: cpu
- dtype: float64
- frames: [0, 40]

Dense Jacobian extraction was not used.

## Call graph and source boundaries
- ASE/MACE batch and neighbor preparation: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/calculators/mace.py:527`
- Frost edge geometry: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/geometry.py:93`
- MACE calculator evaluation: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/calculators/mace.py:593`
- MACE model forward/readout: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/modules/models.py:455`
- MACE interaction block forward: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/modules/blocks.py:763`
- MACE subsequent interaction layers: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/modules/blocks.py:same InteractionBlock.forward for interactions[1:]`
- MACE first interaction/Frost interception: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/mace_patch.py:66`
- layer-1 tensor product and scatter timing: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/mace_patch.py:66`
- Frost affine message substitution: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/mace_patch.py:9`
- Frost JVP action: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/force_error.py:283`
- Frost VJP action: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/force_error.py:295`
- dirty predicate: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/geometry.py:264`
- cache state machine: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/reference_cache.py:71`

## Timer accounting
- **T_exact_total**: CUDA elapsed around calculator.evaluate when CUDA is available, otherwise CPU perf_counter; calculator results are cleared; separate CPU wall counterpart is T_exact_total_cpu_wall
- **T_layer1_TP**: CUDA event or CPU perf_counter around first-layer conv_tp only; nested inside T_layer1_message; excludes conv_tp_weights and cutoff
- **T_layer1_message**: CUDA event or CPU perf_counter from before first-layer conv_tp_weights through mji construction; includes radial tensor-product weights, cutoff multiplication, and conv_tp; nested inside forward
- **T_layer1_aggregation**: CUDA event or CPU perf_counter around first-layer scatter_sum only; nested inside forward and disjoint from T_layer1_message
- **T_rest_forward**: timed model-forward elapsed on the active clock minus complete layer-1 message and scatter events; excludes model_batch() batch/neighbor preparation, and includes layers 2+, readout, and remaining layer-1 work
- **T_backward_force**: CUDA elapsed or CPU perf_counter for explicit autograd.grad of energy with respect to positions; separate from forward
- **T_edge_geometry**: CPU perf_counter around Frost build_edge_geometry_from_ase only; includes ASE neighbor-list construction and MIC geometry
- **T_total_step_wall**: CPU perf_counter enclosing exact evaluation, profiled forward, profiled force forward/backward, and edge geometry; includes host overhead and synchronization
- **microbenchmarks**: each operation has separate CUDA elapsed and CPU wall distributions; JVP/VJP are serial 16-edge action samples; dirty_TP is a 50-percent edge subset
