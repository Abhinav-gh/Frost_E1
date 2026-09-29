# Frost E2 profiler summary

This report measures execution components and action/bookkeeping costs. It is not an end-to-end speedup result.

- theta: 0.791781
- device: cuda
- dtype: float64
- frames: [0]

Dense Jacobian extraction was not used.

## Call graph and source boundaries
- ASE/MACE batch and neighbor preparation: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/calculators/mace.py:527`
- Frost edge geometry: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/geometry.py:93`
- MACE model forward/readout: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/site-packages/mace/modules/models.py:455`
- MACE first interaction/Frost interception: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/contextlib.py:170`
- layer-1 tensor product and scatter timing: `/home/shashankt/.conda/envs/frost_e1/lib/python3.10/contextlib.py:170`
- Frost JVP action: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/force_error.py:283`
- Frost VJP action: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/force_error.py:295`
- dirty predicate: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/geometry.py:264`
- cache state machine: `/home/shashankt/private_storage/projects/Gate_1_test/frost/e1/reference_cache.py:71`
