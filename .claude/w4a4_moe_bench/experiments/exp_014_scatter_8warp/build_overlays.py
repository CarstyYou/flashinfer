#!/usr/bin/env python3
"""Build the locked exp_014 fused 4-warp/8-warp Scatter overlays."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEMORY_ROOT = ROOT.parents[1]
SOURCE = MEMORY_ROOT / "moe_dynamic_kernel_opt.py"
OVERLAY_ROOT = ROOT / "results/overlays"
EXPECTED_BASELINE_SHA256 = (
    "b6e141179794561f2144bdec079b7e109fddfccc2db6ba0f19c22d30ea4b34ca"
)
EXPECTED_CANDIDATE_SHA256 = (
    "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184"
)

REPLACEMENTS = (
    (
        "warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)",
        "warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)",
    ),
    (
        "# Per-warp scatter: each warp scatters its own quadrant\n"
        "            # of sC (64 M-rows x 64 N-cols).",
        "# Per-warp scatter: all eight math warps cover one disjoint\n"
        "            # sC strip (32 M-rows x 64 N-cols).",
    ),
    (
        "if warp_epi_rows > Int32(64):\n                warp_epi_rows = Int32(64)",
        "if warp_epi_rows > Int32(32):\n                warp_epi_rows = Int32(32)",
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def memory_relative(path: Path, memory_root: Path) -> str:
    """Return a checkout-independent path rooted at ``w4a4_moe_bench``."""
    try:
        return path.resolve().relative_to(memory_root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(
            f"overlay identity path escapes w4a4_moe_bench root: {path}"
        ) from error


def apply_replacements(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Apply each locked edit exactly once."""
    result = source
    for old, new in replacements:
        count = result.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one Scatter mapping anchor, got {count}: {old!r}"
            )
        result = result.replace(old, new)
    return result


def build_overlays(
    *,
    source: Path = SOURCE,
    overlay_root: Path = OVERLAY_ROOT,
    memory_root: Path = MEMORY_ROOT,
) -> dict[str, object]:
    baseline_dir = overlay_root / "baseline_4warp_scatter"
    candidate_dir = overlay_root / "candidate_8warp_scatter"
    baseline_path = baseline_dir / "moe_dynamic_kernel.py"
    candidate_path = candidate_dir / "moe_dynamic_kernel.py"

    source_bytes = source.read_bytes()
    input_sha256 = sha256(source_bytes)
    source_text = source_bytes.decode("utf-8")
    if input_sha256 == EXPECTED_BASELINE_SHA256:
        baseline = source_text
        candidate = apply_replacements(baseline, REPLACEMENTS)
    elif input_sha256 == EXPECTED_CANDIDATE_SHA256:
        candidate = source_text
        baseline = apply_replacements(
            candidate, tuple((new, old) for old, new in REPLACEMENTS)
        )
    else:
        raise RuntimeError(
            "locked opt source drift: "
            f"{input_sha256} is neither frozen baseline "
            f"{EXPECTED_BASELINE_SHA256} nor promoted candidate "
            f"{EXPECTED_CANDIDATE_SHA256}"
        )

    if candidate == baseline:
        raise RuntimeError("candidate is identical to baseline")

    baseline_bytes = baseline.encode("utf-8")
    candidate_bytes = candidate.encode("utf-8")
    baseline_sha256 = sha256(baseline_bytes)
    candidate_sha256 = sha256(candidate_bytes)
    if baseline_sha256 != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"reconstructed baseline drift: {baseline_sha256} "
            f"!= {EXPECTED_BASELINE_SHA256}"
        )
    if candidate_sha256 != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            f"reconstructed candidate drift: {candidate_sha256} "
            f"!= {EXPECTED_CANDIDATE_SHA256}"
        )

    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(baseline_bytes)
    candidate_path.write_bytes(candidate_bytes)

    diff = "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="baseline_4warp_scatter/moe_dynamic_kernel.py",
            tofile="candidate_8warp_scatter/moe_dynamic_kernel.py",
            # Zero context avoids unified-diff blank context lines whose single
            # prefix space is unstable under trailing-whitespace hooks.
            n=0,
        )
    )
    changed_lines = [
        line
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    if len(changed_lines) != 10:
        raise RuntimeError(
            f"candidate diff escaped the three locked edits: {len(changed_lines)} lines"
        )

    diff_path = overlay_root / "candidate_8warp_scatter.diff"
    diff_path.write_text(diff, encoding="utf-8")
    identity = {
        "schema": "exp014.kernel-overlay-identity.v1",
        "path_base": "w4a4_moe_bench_root",
        "source": memory_relative(source, memory_root),
        # This identity describes the frozen experiment baseline, even after
        # the accepted candidate has been promoted to the live opt source.
        "source_sha256": baseline_sha256,
        "baseline": memory_relative(baseline_path, memory_root),
        "baseline_sha256": baseline_sha256,
        "candidate": memory_relative(candidate_path, memory_root),
        "candidate_sha256": candidate_sha256,
        "change": {
            "scope": "scatter_sC_to_gmem work mapping only",
            "baseline": "4 effective warps, each 64x64",
            "candidate": "8 effective warps, each 32x64",
            "unchanged": "CTA shape, vector width, REDG count, and all other phases",
        },
        "diff": memory_relative(diff_path, memory_root),
        "changed_diff_lines": changed_lines,
    }
    identity_path = overlay_root / "identity.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def main() -> None:
    identity = build_overlays()
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
