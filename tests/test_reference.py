"""
Tests for src/reference.py: the fixed-priority reference-price selection.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observation import ReferenceCandidate, ReferenceQuality, ReferenceSource
from src.reference import select_reference_price


class TestReferenceSelection(unittest.TestCase):
    def test_tier1_bitget_official_selected_when_available(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=99.5),
            ReferenceCandidate(source=ReferenceSource.LIQUID_PROXY_ESTIMATED, price=101.0),
        ]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.price, 100.0)
        self.assertEqual(sel.source, ReferenceSource.BITGET_OFFICIAL)
        self.assertEqual(sel.quality, ReferenceQuality.OFFICIAL)

    def test_falls_to_tier2_when_tier1_unavailable(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=None, available=False),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=99.5),
        ]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.price, 99.5)
        self.assertEqual(sel.source, ReferenceSource.US_CASH_CLOSE)
        self.assertEqual(sel.quality, ReferenceQuality.OFFICIAL)

    def test_falls_to_tier2_when_tier1_stale(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0, stale=True),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=99.5),
        ]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.source, ReferenceSource.US_CASH_CLOSE)

    def test_falls_to_tier2_when_tier1_price_missing(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=None),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=99.5),
        ]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.source, ReferenceSource.US_CASH_CLOSE)

    def test_falls_to_tier3_liquid_proxy_labelled_estimated(self):
        candidates = [ReferenceCandidate(source=ReferenceSource.LIQUID_PROXY_ESTIMATED, price=102.0)]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.price, 102.0)
        self.assertEqual(sel.source, ReferenceSource.LIQUID_PROXY_ESTIMATED)
        self.assertEqual(sel.quality, ReferenceQuality.ESTIMATED)
        # Never disguised as official, regardless of how it was selected.
        self.assertNotEqual(sel.quality, ReferenceQuality.OFFICIAL)

    def test_no_candidates_usable_returns_unavailable_not_fabricated(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=None),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=50.0, stale=True),
        ]
        sel = select_reference_price(candidates)
        self.assertIsNone(sel.price)
        self.assertEqual(sel.source, ReferenceSource.UNAVAILABLE)
        self.assertEqual(sel.quality, ReferenceQuality.UNAVAILABLE)

    def test_empty_candidate_list_returns_unavailable(self):
        sel = select_reference_price([])
        self.assertIsNone(sel.price)
        self.assertEqual(sel.source, ReferenceSource.UNAVAILABLE)
        self.assertEqual(sel.quality, ReferenceQuality.UNAVAILABLE)

    def test_priority_independent_of_input_order(self):
        candidates = [
            ReferenceCandidate(source=ReferenceSource.LIQUID_PROXY_ESTIMATED, price=102.0),
            ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0),
            ReferenceCandidate(source=ReferenceSource.US_CASH_CLOSE, price=99.5),
        ]
        sel = select_reference_price(candidates)
        self.assertEqual(sel.source, ReferenceSource.BITGET_OFFICIAL)

    def test_unavailable_tier3_falls_through_to_unavailable_result(self):
        candidates = [ReferenceCandidate(source=ReferenceSource.LIQUID_PROXY_ESTIMATED, price=102.0, available=False)]
        sel = select_reference_price(candidates)
        self.assertIsNone(sel.price)
        self.assertEqual(sel.source, ReferenceSource.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
