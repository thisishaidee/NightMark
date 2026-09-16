"""
Tests for src/normalize.py and the src/observation.py data contract.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionCalendar, SessionType
from src.normalize import normalize_observation
from src.observation import RawQuote, ReferenceCandidate, ReferenceQuality, ReferenceSource
from tests.fixtures import SAT_WEEKEND, TUE_RTH

CONTRACT_FIELDS = [
    "timestamp", "symbol", "session",
    "rtoken_bid", "rtoken_ask", "rtoken_mid", "spread_bps", "available_depth",
    "reference_price", "reference_source", "reference_quality",
    "basis", "basis_bps",
]


class TestDataContract(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()
        self.official_ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]

    def test_all_contract_fields_present(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="RNVDA", bid=99.9, ask=100.1)
        obs = normalize_observation(raw, self.official_ref, self.calendar)
        for field in CONTRACT_FIELDS:
            self.assertTrue(hasattr(obs, field), f"missing contract field: {field}")

    def test_session_classification_reused_from_phase1_calendar(self):
        raw = RawQuote(timestamp=SAT_WEEKEND, symbol="RNVDA", bid=99.9, ask=100.1)
        obs = normalize_observation(raw, self.official_ref, self.calendar)
        self.assertEqual(obs.session, SessionType.WEEKEND)

    def test_symbol_and_timestamp_pass_through_unchanged(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="RAAPL", bid=99.9, ask=100.1)
        obs = normalize_observation(raw, self.official_ref, self.calendar)
        self.assertEqual(obs.symbol, "RAAPL")
        self.assertEqual(obs.timestamp, TUE_RTH)


class TestMid(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()
        self.ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]

    def test_normal_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertEqual(obs.rtoken_mid, 100.0)
        self.assertTrue(obs.is_valid)

    def test_equal_bid_ask_is_valid_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=100.0, ask=100.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertEqual(obs.rtoken_mid, 100.0)

    def test_missing_bid_gives_none_mid_and_marks_invalid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=None, ask=101.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.rtoken_mid)
        self.assertFalse(obs.is_valid)
        self.assertEqual(obs.invalid_reason, "invalid_or_missing_rtoken_quote")

    def test_missing_ask_gives_none_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=None)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.rtoken_mid)

    def test_crossed_book_gives_none_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=101.0, ask=99.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.rtoken_mid)
        self.assertFalse(obs.is_valid)

    def test_non_positive_bid_gives_none_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=0.0, ask=101.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.rtoken_mid)

    def test_negative_ask_gives_none_mid(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=-1.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.rtoken_mid)


class TestSpreadBps(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()
        self.ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]

    def test_normal_spread(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        # mid=100, spread=2 -> 2/100*10000 = 200 bps
        self.assertAlmostEqual(obs.spread_bps, 200.0)

    def test_zero_spread(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=100.0, ask=100.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertAlmostEqual(obs.spread_bps, 0.0)

    def test_missing_quote_gives_none_spread(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=None, ask=101.0)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.spread_bps)


class TestDepth(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()
        self.ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]

    def test_depth_is_min_of_both_sides(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0, bid_size=500, ask_size=300)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertEqual(obs.available_depth, 300)

    def test_missing_size_gives_none_depth(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0, bid_size=500, ask_size=None)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.available_depth)

    def test_negative_size_gives_none_depth(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0, bid_size=-5, ask_size=300)
        obs = normalize_observation(raw, self.ref, self.calendar)
        self.assertIsNone(obs.available_depth)


class TestReferenceSourceQualityInObservation(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()

    def test_official_reference_used(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertEqual(obs.reference_source, ReferenceSource.BITGET_OFFICIAL)
        self.assertEqual(obs.reference_quality, ReferenceQuality.OFFICIAL)

    def test_proxy_reference_labelled_estimated_never_official(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        ref = [ReferenceCandidate(source=ReferenceSource.LIQUID_PROXY_ESTIMATED, price=105.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertEqual(obs.reference_source, ReferenceSource.LIQUID_PROXY_ESTIMATED)
        self.assertEqual(obs.reference_quality, ReferenceQuality.ESTIMATED)
        self.assertNotEqual(obs.reference_quality, ReferenceQuality.OFFICIAL)

    def test_no_reference_available_marks_invalid_no_fabrication(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        obs = normalize_observation(raw, [], self.calendar)
        self.assertIsNone(obs.reference_price)
        self.assertEqual(obs.reference_source, ReferenceSource.UNAVAILABLE)
        self.assertFalse(obs.is_valid)
        self.assertEqual(obs.invalid_reason, "no_reference_price_available")


class TestBasis(unittest.TestCase):
    def setUp(self):
        self.calendar = SessionCalendar()

    def test_basis_and_basis_bps_arithmetic(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=101.0, ask=103.0)  # mid=102
        ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertAlmostEqual(obs.basis, 2.0)
        self.assertAlmostEqual(obs.basis_bps, 200.0)  # 2/100*10000

    def test_negative_basis(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=97.0, ask=99.0)  # mid=98
        ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertAlmostEqual(obs.basis, -2.0)
        self.assertAlmostEqual(obs.basis_bps, -200.0)

    def test_missing_reference_gives_none_basis(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        obs = normalize_observation(raw, [], self.calendar)
        self.assertIsNone(obs.basis)
        self.assertIsNone(obs.basis_bps)

    def test_missing_mid_gives_none_basis(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=None, ask=101.0)
        ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=100.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertIsNone(obs.basis)
        self.assertIsNone(obs.basis_bps)

    def test_zero_reference_price_does_not_divide_by_zero(self):
        raw = RawQuote(timestamp=TUE_RTH, symbol="X", bid=99.0, ask=101.0)
        ref = [ReferenceCandidate(source=ReferenceSource.BITGET_OFFICIAL, price=0.0)]
        obs = normalize_observation(raw, ref, self.calendar)
        self.assertIsNone(obs.basis)
        self.assertIsNone(obs.basis_bps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
