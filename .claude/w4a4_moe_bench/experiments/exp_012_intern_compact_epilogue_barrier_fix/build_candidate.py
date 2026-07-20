#!/usr/bin/env python3
"""Build the exp_012 one-change kernel overlay from the locked exp_009 source."""

import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = (
    ROOT.parent
    / "exp_009_intern_stage4_compact_lightcheck"
    / "results/overlays/intern_stage4_compact/moe_dynamic_kernel.py"
)
OUT_DIR = ROOT / "results/overlays/post_quant_barrier"
OUT = OUT_DIR / "moe_dynamic_kernel.py"
EXPECTED_BASELINE_SHA256 = (
    "42ca8d40e18b5d0f001236b09b85cbc0aa30e6010f0954efd538d8b9a2fb57d2"
)

OLD = """                            quant_idx += Int32(
                                self.num_mma_warps * self.num_threads_per_warp
                            )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    # epilog_sync: MMA-only barrier. DMA warp doesn't need to wait
                    # for quant — it only loads B_down into sB (separate buffer).
                    # This allows DMA to prefetch B_down tiles earlier.
                    self.epilog_sync_barrier.arrive_and_wait()
"""

NEW = """                            quant_idx += Int32(
                                self.num_mma_warps * self.num_threads_per_warp
                            )

                        # Finish Q1 before the next compact pass reuses single-stage sC.
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    baseline_bytes = BASELINE.read_bytes()
    observed = sha256(baseline_bytes)
    if observed != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline hash drift: {observed}")
    baseline = baseline_bytes.decode()
    if baseline.count(OLD) != 1:
        raise RuntimeError(
            f"expected one synchronization anchor, got {baseline.count(OLD)}"
        )
    candidate = baseline.replace(OLD, NEW)
    if candidate.count(NEW) != 1 or candidate == baseline:
        raise RuntimeError("candidate replacement did not apply exactly once")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(candidate, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="exp009_intern_adapter/moe_dynamic_kernel.py",
            tofile="exp012_post_quant_barrier/moe_dynamic_kernel.py",
            n=0,
        )
    )
    (OUT_DIR / "candidate.diff").write_text(diff, encoding="utf-8")
    payload = {
        "schema": "exp012.kernel-overlay-identity.v1",
        "baseline": str(BASELINE),
        "baseline_sha256": observed,
        "candidate": str(OUT),
        "candidate_sha256": sha256(candidate.encode()),
        "change": "move the existing post-loop fence/barrier to the end of every epi_m pass",
    }
    (OUT_DIR / "identity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
