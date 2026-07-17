# Attempt 00: harness pair-head oracle mismatch

Candidate A v0 compiled and executed its first M256 replay on 2026-07-17. The
independent-reference and NaN-sentinel checks completed before the harness
stopped at the route/task oracle.

The task descriptor multiset, row counts, expert write rows, tile bases,
task tail, terminal task-head overshoot and publication flag all matched. Only
`pair_head` failed because the original oracle treated it as logical routed
work. In the kernel it is an arm-dependent fixed-size producer-claim counter:
Candidate A uses 9 CTA warps and claims 18 pairs at a time, including one final
out-of-range claim per resident CTA. The observed value `4032` matches
`(ceil(2048/18) + 110) * 18`.

This is a harness-contract failure, not a kernel correctness verdict. The
failed fresh JIT and replay artifacts are retained under this directory. The
canonical rerun uses the corrected, predeclared arm-specific terminal-state
formula.
