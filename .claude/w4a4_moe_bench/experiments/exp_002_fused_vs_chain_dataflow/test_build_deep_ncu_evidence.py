from __future__ import annotations

import unittest

from build_deep_ncu_evidence import interval_union_ns


class IntervalUnionTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(interval_union_ns([]), 0)

    def test_merges_overlap_and_touching(self) -> None:
        self.assertEqual(
            interval_union_ns([(10, 20), (18, 25), (25, 30), (40, 50)]),
            30,
        )


if __name__ == "__main__":
    unittest.main()
