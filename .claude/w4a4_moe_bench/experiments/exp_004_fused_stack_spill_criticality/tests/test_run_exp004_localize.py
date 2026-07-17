from __future__ import annotations

from pathlib import Path

from run_exp004 import parse_args


def test_localization_cli_requires_all_retained_compiler_artifacts() -> None:
    args = parse_args(
        [
            "localize",
            "--baseline-sass",
            "baseline.sass",
            "--up-first-sass",
            "up-first.sass",
            "--baseline-mlir",
            "baseline.mlir",
            "--baseline-ptx",
            "baseline.ptx",
            "--up-first-ptx",
            "up-first.ptx",
            "--source",
            "kernel.py",
        ]
    )
    assert args.command == "localize"
    assert args.baseline_sass == Path("baseline.sass")
    assert args.up_first_sass == Path("up-first.sass")
    assert args.baseline_mlir == Path("baseline.mlir")
    assert args.baseline_ptx == Path("baseline.ptx")
    assert args.up_first_ptx == Path("up-first.ptx")
    assert args.source == Path("kernel.py")
