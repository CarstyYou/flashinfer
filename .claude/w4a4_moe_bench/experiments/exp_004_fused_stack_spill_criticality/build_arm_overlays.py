#!/usr/bin/env python3
"""Build the three pre-registered exp_004 full-module source overlays.

The production file is never edited.  Every transform is anchored to the
audited production SHA and must match an exact number of source occurrences.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TARGET_RELATIVE_PATH = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
EXPECTED_KERNEL_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
    expected: int = 1,
) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


def replace_region(
    text: str,
    *,
    start: str,
    end: str,
    old: str,
    new: str,
    expected: int,
    label: str,
) -> str:
    start_pos = text.find(start)
    if start_pos < 0:
        raise ValueError(f"{label}: start anchor missing")
    end_pos = text.find(end, start_pos + len(start))
    if end_pos < 0:
        raise ValueError(f"{label}: end anchor missing")
    region = text[start_pos:end_pos]
    count = region.count(old)
    if count != expected:
        raise ValueError(
            f"{label}: expected {expected} occurrences of {old!r}, found {count}"
        )
    return text[:start_pos] + region.replace(old, new) + text[end_pos:]


GATED_ACTIVATION_OLD = """                                    if cutlass.const_expr(self.is_gated):
                                        up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = alpha_value * gate_slice[elem_idx]
                                            u = alpha_value * up_slice[elem_idx]
                                            tRS_rD_slice[elem_idx] = (
                                                gated_activation_f32(
                                                    g,
                                                    u,
                                                    activation=self.activation,
                                                    limit=self.swiglu_limit,
                                                    alpha=self.swiglu_alpha,
                                                    beta=self.swiglu_beta,
                                                    fast_math=self.fast_math,
                                                )
                                            )
"""


def gated_activation_new(destination: str) -> str:
    if destination not in {"up_slice", "gate_slice"}:
        raise ValueError(f"unsupported in-place destination: {destination}")
    return f"""                                    if cutlass.const_expr(self.is_gated):
                                        up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                        out_slice = tRS_rD_out[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size({destination})
                                        ):
                                            g = alpha_value * gate_slice[elem_idx]
                                            u = alpha_value * up_slice[elem_idx]
                                            {destination}[elem_idx] = (
                                                gated_activation_f32(
                                                    g,
                                                    u,
                                                    activation=self.activation,
                                                    limit=self.swiglu_limit,
                                                    alpha=self.swiglu_alpha,
                                                    beta=self.swiglu_beta,
                                                    fast_math=self.fast_math,
                                                )
                                            )
                                        out_slice.store(
                                            {destination}.load().to(cutlass.BFloat16)
                                        )
"""


ACTIVATION_CONVERT_OLD = """                            acc_vec = tRS_rD.load()
                            acc_vec = acc_vec.to(cutlass.BFloat16)
                            tRS_rD_out.store(acc_vec)
                            cute.copy(
"""

ACTIVATION_CONVERT_NEW = """                            if cutlass.const_expr(not self.is_gated):
                                acc_vec = tRS_rD.load()
                                acc_vec = acc_vec.to(cutlass.BFloat16)
                                tRS_rD_out.store(acc_vec)
                            cute.copy(
"""


def build_activation_in_place(source: str, destination: str) -> str:
    arm = f"activation_in_place_{destination.removesuffix('_slice')}"
    transformed = replace_exact(
        source,
        GATED_ACTIVATION_OLD,
        gated_activation_new(destination),
        label=f"{arm}: gated activation",
    )
    # The same load/convert/store sequence appears again in FC2, but FC2 has an
    # intervening epi_buffer assignment.  This longer anchor is therefore
    # unique to FC1 and leaves the FC2 epilogue byte-for-byte unchanged.
    first = transformed.find(ACTIVATION_CONVERT_OLD)
    if first < 0 or transformed.count(ACTIVATION_CONVERT_OLD) != 1:
        raise ValueError(f"{arm}: expected one FC1 conversion block")
    transformed = (
        transformed[:first]
        + ACTIVATION_CONVERT_NEW
        + transformed[first + len(ACTIVATION_CONVERT_OLD) :]
    )
    return transformed


def build_up_first(source: str) -> str:
    transformed = replace_region(
        source,
        start="                    # Gate GEMM (inlined to avoid @cute.jit pass-by-value for acc)\n",
        end="                    # Signal FC1 gate/only completion before the DMA warp\n",
        old="gate_acc",
        new="up_acc",
        expected=5,
        label="up_first: first consumer pass",
    )
    transformed = replace_region(
        transformed,
        start="                    if cutlass.const_expr(self.is_gated):\n                        # Up GEMM (inlined, same pattern)\n",
        end="                    # Activation + quant into sA\n",
        old="up_acc",
        new="gate_acc",
        expected=5,
        label="up_first: second consumer pass",
    )
    transformed = replace_region(
        transformed,
        start="                    # ---- FC1 gate/only pass ----\n",
        end="                    # Wait for the MMA warps to finish the FC1 gate/only pass\n",
        old="tBgB_w13_gate_nk",
        new="tBgB_w13_up_nk",
        expected=1,
        label="up_first: first producer B",
    )
    transformed = replace_region(
        transformed,
        start="                    # ---- FC1 gate/only pass ----\n",
        end="                    # Wait for the MMA warps to finish the FC1 gate/only pass\n",
        old="tBgSFB_w13_gate_nk",
        new="tBgSFB_w13_up_nk",
        expected=1,
        label="up_first: first producer SFB",
    )
    transformed = replace_region(
        transformed,
        start="                        # ---- FC1 up pass ----\n",
        end="                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        old="tBgB_w13_up_nk",
        new="tBgB_w13_gate_nk",
        expected=1,
        label="up_first: second producer B",
    )
    transformed = replace_region(
        transformed,
        start="                        # ---- FC1 up pass ----\n",
        end="                    # ---- FC2 B_down loads: continuous pipeline ----\n",
        old="tBgSFB_w13_up_nk",
        new="tBgSFB_w13_gate_nk",
        expected=1,
        label="up_first: second producer SFB",
    )
    return transformed


def build_overlays(source: str) -> dict[str, str]:
    overlays = {
        "activation_in_place_up": build_activation_in_place(source, "up_slice"),
        "activation_in_place_gate": build_activation_in_place(source, "gate_slice"),
        "up_first_attribution": build_up_first(source),
    }
    for arm, text in overlays.items():
        if text == source:
            raise ValueError(f"{arm}: transform produced a byte-identical overlay")
        ast.parse(text, filename=f"{arm}.py")
    return overlays


def write_overlays(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    if sha256_file(source_path) != EXPECTED_KERNEL_SHA256:
        raise ValueError("production kernel SHA-256 drift")
    source = source_path.read_text()
    overlays = build_overlays(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    identities: dict[str, Any] = {}
    for arm, text in overlays.items():
        path = output_dir / f"{arm}.py"
        path.write_text(text)
        diff = "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile="production/moe_dynamic_kernel.py",
                tofile=f"generated/{arm}.py",
            )
        )
        diff_path = output_dir / f"{arm}.patch"
        diff_path.write_text(diff)
        identities[arm] = {
            "overlay": path.name,
            "overlay_sha256": sha256_file(path),
            "patch": diff_path.name,
            "patch_sha256": sha256_file(diff_path),
        }
    manifest = {
        "schema": "exp004.generated-overlays.v1",
        "production": {
            "path": str(source_path),
            "sha256": EXPECTED_KERNEL_SHA256,
        },
        "arms": identities,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_overlays(
        args.flashinfer_root.resolve() / TARGET_RELATIVE_PATH,
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
