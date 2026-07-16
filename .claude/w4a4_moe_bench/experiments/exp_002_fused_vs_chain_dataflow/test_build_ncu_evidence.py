from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from build_ncu_evidence import RAW_METRICS, parse_native_raw


def write_native_csv(
    path: Path, *, unit_override: tuple[str, str] | None = None
) -> None:
    headers = ["ID", "Kernel Name", *RAW_METRICS.values()]
    units = ["", ""]
    values = ["0", "range"]
    for metric in RAW_METRICS.values():
        if "duration" in metric:
            unit, value = "ms", "2"
        elif "bytes" in metric:
            unit, value = "Mbyte", "3"
        elif "sectors" in metric:
            unit, value = "sector", "4"
        elif "inst" in metric:
            unit, value = "inst", "5"
        else:
            unit, value = "", "6"
        if unit_override and metric == unit_override[0]:
            unit = unit_override[1]
        units.append(unit)
        values.append(value)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows((headers, units, values))


class ParseNativeRawTest(unittest.TestCase):
    def test_scales_units_and_derives_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native_raw.csv"
            write_native_csv(path)
            metrics, units = parse_native_raw(path)
        self.assertEqual(metrics["profiler_range_duration_ns"], 2e6)
        self.assertEqual(metrics["dram_read_bytes"], 3e6)
        self.assertEqual(metrics["dram_total_authority_bytes"], 3e6)
        self.assertEqual(metrics["dram_component_sum_bytes"], 6e6)
        self.assertEqual(metrics["l2_read_sectors"], 4)
        self.assertEqual(metrics["l2_total_authority_bytes"], 128)
        self.assertEqual(metrics["l2_classified_component_sectors"], 20)
        self.assertEqual(metrics["lsu_global_t_stage_footprint_bytes"], 15e6)
        self.assertEqual(metrics["tma_global_interface_bytes"], 9e6)
        self.assertEqual(metrics["local_total_footprint_bytes"], 6e6)
        self.assertEqual(metrics["dynamic_warp_instructions"], 5)
        self.assertEqual(units["dram_read_bytes"], "Mbyte")

    def test_rejects_unknown_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native_raw.csv"
            write_native_csv(
                path,
                unit_override=("dram__bytes_op_read.sum", "MiB"),
            )
            with self.assertRaisesRegex(ValueError, "unsupported NCU unit"):
                parse_native_raw(path)


if __name__ == "__main__":
    unittest.main()
