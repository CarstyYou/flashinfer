#!/usr/bin/env python3
"""Build the immutable exp_005 baseline and Candidate-A source overlays."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple


PRODUCTION_SOURCE_REL = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
PRODUCTION_SHA256 = "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"

BASELINE_NAME = "baseline_4warp"
CANDIDATE_NAME = "candidate_8warp_serial_v0"

# Candidate A v0 is deliberately restricted to these two exact byte transforms.
TRANSFORMS: Sequence[Tuple[bytes, bytes, str]] = (
    (
        b"        self.num_mma_warps = 4\n",
        b"        self.num_mma_warps = 8\n",
        "compute warps: 4 -> 8 (TMA warp follows num_mma_warps)",
    ),
    (
        b"        atom_layout = cute.make_layout((2, 2, 1))\n",
        b"        atom_layout = cute.make_layout((4, 2, 1))\n",
        "tiled-MMA atom layout: (2, 2, 1) -> (4, 2, 1)",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_root() -> Path:
    """Resolve the FlashInfer repository root from this experiment script."""

    return Path(__file__).resolve().parents[4]


def _parse_python(source: bytes, filename: str) -> None:
    ast.parse(source.decode("utf-8"), filename=filename)


def _apply_exact_transforms(
    production: bytes,
) -> Tuple[bytes, Sequence[Dict[str, object]]]:
    candidate = production
    records = []
    for before, after, description in TRANSFORMS:
        match_count = candidate.count(before)
        if match_count != 1:
            raise RuntimeError(
                "exact transform {!r} expected one match, found {}".format(
                    description, match_count
                )
            )
        candidate = candidate.replace(before, after, 1)
        records.append(
            {
                "description": description,
                "before": before.decode("utf-8").rstrip("\n"),
                "after": after.decode("utf-8").rstrip("\n"),
                "match_count": match_count,
            }
        )
    return candidate, records


def _unified_diff(production: bytes, candidate: bytes) -> str:
    return "".join(
        difflib.unified_diff(
            production.decode("utf-8").splitlines(keepends=True),
            candidate.decode("utf-8").splitlines(keepends=True),
            fromfile="production/moe_dynamic_kernel.py",
            tofile="{}/moe_dynamic_kernel.py".format(CANDIDATE_NAME),
        )
    )


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_overlays(source_path: Path, output_dir: Path) -> Dict[str, object]:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    production = source_path.read_bytes()
    production_sha256 = _sha256(production)
    if production_sha256 != PRODUCTION_SHA256:
        raise RuntimeError(
            "production source hash drift: expected {}, got {} ({})".format(
                PRODUCTION_SHA256, production_sha256, source_path
            )
        )

    baseline = production
    candidate, transform_records = _apply_exact_transforms(production)

    _parse_python(baseline, "{}/moe_dynamic_kernel.py".format(BASELINE_NAME))
    _parse_python(candidate, "{}/moe_dynamic_kernel.py".format(CANDIDATE_NAME))

    if baseline != production:
        raise AssertionError("baseline overlay is not byte-identical to production")

    baseline_path = output_dir / BASELINE_NAME / "moe_dynamic_kernel.py"
    candidate_path = output_dir / CANDIDATE_NAME / "moe_dynamic_kernel.py"
    diff_path = output_dir / "{}.diff".format(CANDIDATE_NAME)
    identity_path = output_dir / "identity.json"

    _write(baseline_path, baseline)
    _write(candidate_path, candidate)
    diff_bytes = _unified_diff(production, candidate).encode("utf-8")
    _write(diff_path, diff_bytes)

    identity = {
        "schema_version": 1,
        "experiment": "exp_005_8warp_spill_reduction",
        "candidate_version": "v0",
        "production": {
            "path": str(source_path),
            "repository_relative_path": PRODUCTION_SOURCE_REL.as_posix(),
            "sha256": production_sha256,
            "locked_sha256": PRODUCTION_SHA256,
        },
        "arms": {
            BASELINE_NAME: {
                "path": str(baseline_path),
                "sha256": _sha256(baseline),
                "size_bytes": len(baseline),
                "byte_identical_to_production": baseline == production,
                "ast_parse": "pass",
            },
            CANDIDATE_NAME: {
                "path": str(candidate_path),
                "sha256": _sha256(candidate),
                "size_bytes": len(candidate),
                "byte_identical_to_production": candidate == production,
                "ast_parse": "pass",
                "transforms": transform_records,
            },
        },
        "unified_diff": {
            "path": str(diff_path),
            "sha256": _sha256(diff_bytes),
            "size_bytes": len(diff_bytes),
        },
    }
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write(identity_path, identity_bytes)
    return identity


def parse_args() -> argparse.Namespace:
    root = repository_root()
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / PRODUCTION_SOURCE_REL,
        help="production moe_dynamic_kernel.py (hash-locked)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir / "results" / "overlays",
        help="overlay artifact directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = build_overlays(args.source, args.output_dir)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
