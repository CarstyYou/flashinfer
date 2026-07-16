# Exp 003 IKET cadence breakdown

- Evidence scope: 3 selected-CTA captures; timestamps are `raw timestamp units`.
- Parallel MMA-warp intervals were analyzed independently and were never added.
- Coverage gate: **fail**.
- Decision: **inconclusive**.
- Trace-capacity gate: **pass** for every target warp; NativeDump uses a 16-byte header, `bytesWritten == 16 + len(raw_data)*4`, capacity `16 + maxTsCntPerWarp*8`, and <90% utilization.
- Decoded-event and binary semantic OMMA gates: **pass**.
- Static binary formal-dominance eligibility: **missing or failed**; diagnostics remain available.
- Above-calibration wait/barrier PC/SASS closure: **pass**.

| Bucket | Weighted point | Bootstrap 95% interval |
|---|---:|---:|
| tensor | missing | 6.17% – 7.41% |
| planned | missing | 16.84% – 17.73% |
| starvation | missing | 1.27% – 1.35% |
| orchestration | missing | 0.31% – 0.32% |
| unclassified | missing | 73.90% – 74.75% |

Coverage is defined per population stratum:

- `early|full|slices=1`: 3 complete tasks / 3 captures — fail
- `early|partial|slices=1`: 4 complete tasks / 3 captures — fail
- `steady|full|slices=1`: 28 complete tasks / 3 captures — pass
- `steady|partial|slices=1`: 28 complete tasks / 3 captures — pass
- `tail|full|slices=1`: 4 complete tasks / 2 captures — fail
- `tail|partial|slices=1`: 3 complete tasks / 2 captures — fail

The estimate is an instrumented sampled-warp diagnostic. It is not production latency and is not an NCU active-cycle denominator.
Top-level fixed phases are reported per warp separately and are not assigned to an arbitrary dynamic task stratum.
