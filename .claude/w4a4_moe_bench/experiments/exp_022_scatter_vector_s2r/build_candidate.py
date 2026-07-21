#!/usr/bin/env python3
"""Build the identity-locked exp_022 vector-S2R overlay.

The only kernel change is the Scatter sC read width: eight scalar BF16 loads
become one 128-bit shared-memory load from the address returned by the existing
K_SW128 layout.  The script validates every warp/lane/loop address before it
writes the candidate.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent.parent / "moe_dynamic_kernel_opt.py"
OUT_DIR = ROOT / "results/overlays/candidate_vector_s2r_v0"
OUT = OUT_DIR / "moe_dynamic_kernel.py"
EXPECTED_BASELINE_SHA256 = (
    "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
)
BASELINE_GIT_BLOB = "0aa1cf9bb4b345c9438e3a5f0c62b56c1f9b046b"


IMPORT_OLD = """    get_ptr_as_int64,
    st_global_f32,
"""

IMPORT_NEW = """    get_ptr_as_int64,
    get_smem_ptr_as_int32,
    st_global_f32,
"""


HELPER_ANCHOR = """@dsl_user_op
def _ld_global_u64(addr, *, loc=None, ip=None):
"""


HELPER = r'''@dsl_user_op
def load_shared_bf16x8_to_f32x8(addr: Int32, *, loc=None, ip=None):
    """Load one aligned BF16x8 vector with exactly one 128-bit S2R."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 p0, p1, p2, p3;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            ld.shared.v4.u32 {p0, p1, p2, p3}, [$8];
            mov.b32 {b0, b1}, p0;
            mov.b32 {b2, b3}, p1;
            mov.b32 {b4, b5}, p2;
            mov.b32 {b6, b7}, p3;
            cvt.f32.bf16 $0, b0;
            cvt.f32.bf16 $1, b1;
            cvt.f32.bf16 $2, b2;
            cvt.f32.bf16 $3, b3;
            cvt.f32.bf16 $4, b4;
            cvt.f32.bf16 $5, b5;
            cvt.f32.bf16 $6, b6;
            cvt.f32.bf16 $7, b7;
        }
        """,
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = []
    for idx in range(8):
        value = llvm.extractvalue(T.f32(), result, [idx], loc=loc, ip=ip)
        values.append(cutlass.Float32(value))
    return tuple(values)


'''


SCALAR_LOADS = """                sc_v0 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col,
                        epi_buffer,
                    ]
                )
                sc_v1 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(1),
                        epi_buffer,
                    ]
                )
                sc_v2 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(2),
                        epi_buffer,
                    ]
                )
                sc_v3 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(3),
                        epi_buffer,
                    ]
                )
                sc_v4 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(4),
                        epi_buffer,
                    ]
                )
                sc_v5 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(5),
                        epi_buffer,
                    ]
                )
                sc_v6 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(6),
                        epi_buffer,
                    ]
                )
                sc_v7 = cutlass.Float32(
                    sC[
                        warp_m_base + local_row,
                        local_col + Int32(7),
                        epi_buffer,
                    ]
                )
"""


VECTOR_LOAD = """                # Preserve the K_SW128 address transform: compute the
                # unswizzled outer offset through sC.layout, then explicitly
                # apply S<3,4,3> in BF16 element units before stripping the
                # SMEM pointer metadata.  A raw pointer does not retain CuTe's
                # swizzle transform.
                sc_element_offset = Int32(
                    sC.layout(
                        (
                            warp_m_base + local_row,
                            local_col,
                            epi_buffer,
                        )
                    )
                )
                sc_element_offset = sc_element_offset ^ (
                    (sc_element_offset & Int32(0x1C0)) >> Int32(3)
                )
                sc_smem_addr = get_smem_ptr_as_int32(
                    sC,
                    sc_element_offset,
                )
                (
                    sc_v0,
                    sc_v1,
                    sc_v2,
                    sc_v3,
                    sc_v4,
                    sc_v5,
                    sc_v6,
                    sc_v7,
                ) = load_shared_bf16x8_to_f32x8(sc_smem_addr)
"""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def swizzle_3_4_3(element_offset):
    """Apply CuTe S<3,4,3> in byte-address units to a BF16 offset."""
    byte_offset = element_offset * 2
    physical_byte_offset = byte_offset ^ ((byte_offset & 0x380) >> 3)
    if physical_byte_offset % 2:
        raise RuntimeError("BF16 swizzle produced an odd byte address")
    return physical_byte_offset // 2


def unswizzled_k_sw128_offset(row, col, stage=0):
    """BF16 element offset for tiled K_SW128 M128xN128x1.

    K_SW128's atom is (M8,N64):(64,1).  tile_to_shape uses M tile,
    then N tile, then stage order for row-major C.
    """
    if not (0 <= row < 128 and 0 <= col < 128 and stage == 0):
        raise ValueError((row, col, stage))
    atom_m, atom_n = row // 8, col // 64
    in_atom = (row % 8) * 64 + (col % 64)
    return in_atom + atom_m * 512 + atom_n * (16 * 512) + stage * 16384


def physical_offset(row, col, stage=0):
    return swizzle_3_4_3(unswizzled_k_sw128_offset(row, col, stage))


def enumerate_scatter_addresses():
    # Independent known answer for K_SW128 BF16: logical (row=1,col=0)
    # starts at byte 128; S<3,4,3> maps it to byte 144 (element 72).
    if physical_offset(1, 0) != 72:
        raise RuntimeError("K_SW128 known-answer check failed for (row=1,col=0)")
    checked_count = 0
    record_digest = hashlib.sha256()
    first_record = None
    last_record = None
    physical_elements = set()
    segments = set()
    for warp in range(8):
        warp_m_base = (warp >> 1) * 32
        warp_n_base = (warp & 1) * 64
        for lane in range(32):
            for loop in range(8):
                vec_idx = lane + loop * 32
                local_row = vec_idx // 8
                local_vec_col = vec_idx % 8
                row = warp_m_base + local_row
                col = warp_n_base + local_vec_col * 8
                logical = [(row, col + i) for i in range(8)]
                physical = [physical_offset(r, c) for r, c in logical]
                if physical != list(range(physical[0], physical[0] + 8)):
                    raise RuntimeError(
                        f"non-contiguous swizzled vector: warp={warp} lane={lane} "
                        f"loop={loop} logical={logical} physical={physical}"
                    )
                if (physical[0] * 2) % 16:
                    raise RuntimeError(
                        f"unaligned BF16x8: warp={warp} lane={lane} loop={loop} "
                        f"byte={physical[0] * 2}"
                    )
                if any(r != row for r, _ in logical) or logical[-1][1] // 8 != col // 8:
                    raise RuntimeError("vector crosses a logical row/vector boundary")
                segment = tuple(physical)
                if segment in segments:
                    raise RuntimeError(f"duplicate physical segment: {segment}")
                segments.add(segment)
                physical_elements.update(physical)
                record = {
                    "warp": warp,
                    "lane": lane,
                    "loop": loop,
                    "buffer": 0,
                    "logical_row": row,
                    "logical_col_begin": col,
                    "physical_element_begin": physical[0],
                    "physical_byte_begin": physical[0] * 2,
                }
                encoded = json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                ).encode()
                record_digest.update(encoded)
                record_digest.update(b"\n")
                first_record = record if first_record is None else first_record
                last_record = record
                checked_count += 1
    if checked_count != 8 * 32 * 8:
        raise RuntimeError("scatter vector count drift")
    if physical_elements != set(range(128 * 128)):
        missing = sorted(set(range(128 * 128)) - physical_elements)[:16]
        extra = sorted(physical_elements - set(range(128 * 128)))[:16]
        raise RuntimeError(f"physical coverage drift: missing={missing}, extra={extra}")
    return {
        "schema": "exp022.k_sw128_address_enumeration.v1",
        "layout": {
            "logical_shape": [128, 128, 1],
            "element": "BF16",
            "atom": "K_SW128=(M8,N64):(64,1)+S<3,4,3>",
            "base_alignment_bytes": 1024,
        },
        "vector_width_bits": 128,
        "checked_count": checked_count,
        "canonical_record_stream_sha256": record_digest.hexdigest(),
        "unique_physical_elements": len(physical_elements),
        "all_segments_contiguous": True,
        "all_segments_16b_aligned": True,
        "complete_unique_tile_coverage": True,
        "examples": {"first": first_record, "last": last_record},
    }


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def read_locked_baseline():
    live_bytes = BASELINE.read_bytes()
    if sha256_bytes(live_bytes) == EXPECTED_BASELINE_SHA256:
        return live_bytes, "live pre-integration source"
    repo = ROOT.parents[3]
    baseline_bytes = subprocess.check_output(
        ["git", "cat-file", "blob", BASELINE_GIT_BLOB], cwd=str(repo)
    )
    if sha256_bytes(baseline_bytes) != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("locked baseline git blob SHA drift")
    return baseline_bytes, "locked git blob " + BASELINE_GIT_BLOB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    baseline_bytes, baseline_origin = read_locked_baseline()
    baseline_sha = sha256_bytes(baseline_bytes)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"baseline SHA drift: expected {EXPECTED_BASELINE_SHA256}, got {baseline_sha}"
        )

    address_evidence = enumerate_scatter_addresses()
    source = baseline_bytes.decode()
    source = replace_once(source, IMPORT_OLD, IMPORT_NEW, "SMEM pointer import")
    source = replace_once(
        source, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "vector helper"
    )
    source = replace_once(source, SCALAR_LOADS, VECTOR_LOAD, "Scatter sC load")

    candidate_bytes = source.encode()
    candidate_sha = sha256_bytes(candidate_bytes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "results/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        OUT,
        OUT_DIR / "candidate.diff",
        OUT_DIR / "identity.json",
        raw_dir / "swizzle_address_enumeration.json",
    ]
    if not args.force and any(path.exists() for path in targets):
        existing = ", ".join(str(path) for path in targets if path.exists())
        raise RuntimeError(f"refusing to overwrite existing output: {existing}")

    OUT.write_bytes(candidate_bytes)
    diff = "".join(
        difflib.unified_diff(
            baseline_bytes.decode().splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile="moe_dynamic_kernel_opt.py",
            tofile="candidate_vector_s2r_v0/moe_dynamic_kernel.py",
        )
    )
    (OUT_DIR / "candidate.diff").write_text(diff)
    identity = {
        "schema": "exp022.candidate-identity.v1",
        "baseline": str(BASELINE),
        "baseline_origin": baseline_origin,
        "baseline_sha256": baseline_sha,
        "candidate": str(OUT),
        "candidate_sha256": candidate_sha,
        "changed_bundle": "Scatter 8xLDS.U16 -> one ld.shared.v4.u32",
        "address_evidence": str(raw_dir / "swizzle_address_enumeration.json"),
    }
    (OUT_DIR / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (raw_dir / "swizzle_address_enumeration.json").write_text(
        json.dumps(address_evidence, indent=2) + "\n"
    )
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
