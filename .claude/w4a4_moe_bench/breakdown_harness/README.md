# W4A4 MoE Breakdown Harness

This directory owns only the stable pieces reused by multiple experiments.
An experiment may and usually should keep a custom runner, adapter, probe, and
configuration under its own `exp_xxx/` directory.

The dependency direction is:

```text
breakdown_harness  <-  exp_xxx custom harness
```

An experiment must not import executable Python or shell code from another
experiment. Such an import is an extraction trigger, not a reason to move the
whole imported file: extract only the stable shared portion and leave the
experiment identity, constants, candidate, overlay, and one-off probe local.

## Ownership

- `case.py`: canonical cases, persisted fixtures, NVFP4 weights, oracle, and
  output comparison.
- `artifacts.py`: content identity and atomic serialization; generated files
  still belong to the calling experiment.
- `backends/`: reusable implementation adapters after at least two real uses.
- `fragments/`: reviewed, parameterized source or instrumentation fragments
  after at least two real uses.
- `test_contract.py`: CPU-safe shared-contract tests.

The current backend slice contains the shared CuteDSL W4A4 launch/workspace
primitives and the SGLang Triton FP8 launch/oracle primitives. The Eric stage-4
constructor transformation is the first reviewed reusable fragment. Their CLI,
environment pins, comparison protocol, and result schema remain in exp001,
exp005, exp009, and exp018.

Generated overlays, binaries, traces, NCU/IKET reports, CSV, and JSON evidence
belong under the producing experiment's `results/`. Another experiment may
reuse immutable evidence by locator plus SHA256; generated files are never
copied into this source directory. A non-versioned cache, when useful, must be
gitignored and content-addressed.

Add a shared module only when a second real use proves a stable boundary. Do
not prebuild modules merely to complete a framework shape.

Closed historical experiments are not rewritten solely to satisfy the new
layout. Before a historical path is reactivated, remove executable imports from
other experiments and extract only the stable leaf it needs. Shared profiler
capture/extraction is intentionally deferred until the next real profiling
task proves its boundary.
