"""
Tests for ml_rag_adapter.py

Tests all 6 event types, metadata preservation, patient ID exclusion,
retrieval query content, and batch loading. NO LLM calls.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_rag_adapter import (
    MLClinicalEvent,
    MLRAGResult,
    transform_query,
    _safe_float,
    _safe_int,
    load_events_from_csv,
    _QUERY_TEMPLATES,
)


class TestSafeConversions(unittest.TestCase):
    def test_safe_float_normal(self):
        self.assertEqual(_safe_float("77.0"), 77.0)

    def test_safe_float_empty(self):
        self.assertEqual(_safe_float(""), 0.0)

    def test_safe_float_none(self):
        self.assertEqual(_safe_float(None), 0.0)

    def test_safe_float_invalid(self):
        self.assertEqual(_safe_float("abc", 5.0), 5.0)

    def test_safe_int_normal(self):
        self.assertEqual(_safe_int("9"), 9)

    def test_safe_int_float_string(self):
        self.assertEqual(_safe_int("9.0"), 9)


class TestQueryTransformation(unittest.TestCase):
    """Test transform_query for all 6 event types."""

    def _make_event(self, event_type, baseline=77.0, observed=102.0,
                    deviation=25.0, duration=9.0, **kwargs):
        return MLClinicalEvent(
            event_id="EVT-TEST",
            subject_id="9999999",
            event_type=event_type,
            clinical_query=f"Original narrative query about {event_type}",
            personal_baseline_hr=baseline,
            observed_mean_hr=observed,
            deviation_bpm=deviation,
            duration_minutes=duration,
            **kwargs,
        )

    def test_all_six_event_types_produce_nonempty_query(self):
        for etype in _QUERY_TEMPLATES:
            event = self._make_event(etype)
            result = transform_query(event)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 20, f"Query too short for {etype}")

    def test_sustained_elevation_contains_tachycardia(self):
        event = self._make_event("sustained_elevation")
        q = transform_query(event)
        self.assertIn("tachycardia", q.lower())

    def test_sustained_reduction_contains_bradycardia(self):
        event = self._make_event("sustained_reduction", observed=59.0, deviation=-18.0)
        q = transform_query(event)
        self.assertIn("bradycardia", q.lower())

    def test_sudden_change_contains_sudden(self):
        event = self._make_event("sudden_change")
        q = transform_query(event)
        self.assertIn("sudden", q.lower())

    def test_recurring_elevation_contains_recurrent(self):
        event = self._make_event("recurring_elevation")
        q = transform_query(event)
        self.assertIn("recurrent", q.lower())

    def test_recurring_reduction_contains_bradycardia(self):
        event = self._make_event("recurring_reduction", observed=62.0, deviation=-15.0)
        q = transform_query(event)
        self.assertIn("bradycardia", q.lower())

    def test_unusual_variability_contains_arrhythmia(self):
        event = self._make_event("unusual_variability", observed=110.0, deviation=37.0)
        q = transform_query(event)
        self.assertIn("arrhythmia", q.lower())

    def test_all_queries_contain_heart_failure(self):
        for etype in _QUERY_TEMPLATES:
            event = self._make_event(etype)
            q = transform_query(event)
            self.assertIn("heart failure", q.lower(),
                          f"Missing 'heart failure' in {etype} query")

    def test_clinical_signal_appended(self):
        event = self._make_event("sustained_elevation", baseline=77.0,
                                 observed=102.0, deviation=25.0, duration=9.0)
        q = transform_query(event)
        self.assertIn("102 bpm", q)
        self.assertIn("77 bpm", q)
        self.assertIn("25 bpm", q)
        self.assertIn("9 minutes", q)

    def test_no_patient_identifiers_in_query(self):
        event = self._make_event("sustained_elevation")
        event.subject_id = "2022484408"
        q = transform_query(event)
        self.assertNotIn("2022484408", q)
        self.assertNotIn(event.subject_id, q)

    def test_no_timestamps_in_query(self):
        event = self._make_event("sustained_elevation")
        event.start_time = "2016-04-01 07:54:00"
        q = transform_query(event)
        self.assertNotIn("2016", q)
        self.assertNotIn("07:54", q)

    def test_unknown_event_type_returns_original(self):
        event = self._make_event("unknown_type")
        q = transform_query(event)
        self.assertEqual(q, event.clinical_query)

    def test_zero_baseline_omits_details(self):
        event = self._make_event("sustained_elevation", baseline=0, observed=0,
                                 deviation=0, duration=0)
        q = transform_query(event)
        # Should still have the base template
        self.assertIn("tachycardia", q.lower())
        # Should NOT have bpm details
        self.assertNotIn("bpm", q)


class TestMLClinicalEvent(unittest.TestCase):
    def test_dataclass_creation(self):
        event = MLClinicalEvent(
            event_id="EVT-0001",
            subject_id="2022484408",
            event_type="sustained_elevation",
            clinical_query="Test query",
        )
        self.assertEqual(event.event_id, "EVT-0001")
        self.assertEqual(event.event_type, "sustained_elevation")
        self.assertEqual(event.personal_baseline_hr, 0.0)

    def test_optional_fields_default(self):
        event = MLClinicalEvent(
            event_id="EVT-0002",
            subject_id="111",
            event_type="sustained_reduction",
            clinical_query="Query",
        )
        self.assertEqual(event.start_time, "")
        self.assertEqual(event.duration_minutes, 0.0)


class TestCSVLoading(unittest.TestCase):
    """Test that load_events_from_csv loads real data correctly."""

    @classmethod
    def setUpClass(cls):
        cls.events = load_events_from_csv()

    def test_loads_all_events(self):
        self.assertEqual(len(self.events), 12302)

    def test_all_six_types_present(self):
        types = {e.event_type for e in self.events}
        expected = {
            "sustained_elevation", "sustained_reduction",
            "sudden_change", "recurring_elevation",
            "recurring_reduction", "unusual_variability",
        }
        self.assertEqual(types, expected)

    def test_event_metadata_populated(self):
        evt = self.events[0]
        self.assertGreater(evt.personal_baseline_hr, 0)
        self.assertGreater(evt.observed_mean_hr, 0)

    def test_event_ids_unique(self):
        ids = [e.event_id for e in self.events]
        self.assertEqual(len(ids), len(set(ids)))

    def test_transform_works_on_all_events(self):
        failures = []
        for e in self.events:
            q = transform_query(e)
            if not q or len(q) < 20:
                failures.append(e.event_id)
        self.assertEqual(failures, [],
                         f"Transform failed for {len(failures)} events")


class TestMLRAGResult(unittest.TestCase):
    def test_dataclass_creation(self):
        result = MLRAGResult(
            event_id="EVT-0001",
            subject_id="999",
            event_type="sustained_elevation",
            original_query="original",
            retrieval_query="adapted",
            integration_status="SUCCESS",
        )
        self.assertEqual(result.integration_status, "SUCCESS")


if __name__ == "__main__":
    unittest.main()
