# Frost Project Board

E1 Pilot — Methodology Failure, Not Scientific Failure

The initial MACE atom-level stale-state proxy successfully validated the trajectory/cache instrumentation and geometric redundancy measurements, but its force-error measurement is not a valid representation of Frost's edge-local cache. It freezes atomic coordinates rather than selectively reusing edge-local geometric tensors, introducing artificial nonphysical strain. The resulting FAIL verdict is therefore not evidence against Frost. Next step is to implement a minimal edge-level differential harness inside MACE, validating exact-vs-stale force evaluation for zeroth- and mandatory first-order cache representations. E2 cost decomposition may proceed in parallel.

