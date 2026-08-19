import unittest
from confidence_policy import (
    EvidenceInput,
    calculate_confidence,
    load_config_from_json,
)


class TestConfidencePolicy(unittest.TestCase):
    def test_high(self):
        r = calculate_confidence(EvidenceInput(.90, .95, .90, 1.0, 3))
        self.assertEqual((r.level, r.decision), ("high", "answer"))

    def test_medium_partial(self):
        r = calculate_confidence(EvidenceInput(.80, .60, .70, 1.0, 1))
        self.assertEqual((r.level, r.decision), ("medium", "caution"))

    def test_empty_retrieval(self):
        r = calculate_confidence(EvidenceInput(.90, .90, .90, 1.0, 0))
        self.assertEqual((r.level, r.decision), ("low", "refuse"))

    def test_low_coverage(self):
        r = calculate_confidence(EvidenceInput(.95, .20, .90, 1.0, 3))
        self.assertEqual((r.level, r.decision), ("low", "refuse"))

    def test_one_chunk_cannot_be_high(self):
        r = calculate_confidence(EvidenceInput(.99, .95, .99, 1.0, 1))
        self.assertEqual((r.level, r.decision), ("medium", "caution"))

    def test_conflicting_sources(self):
        r = calculate_confidence(EvidenceInput(.95, .95, .95, 1.0, 4, True))
        self.assertEqual((r.level, r.decision), ("medium", "caution"))

    def test_high_score_not_enough(self):
        r = calculate_confidence(EvidenceInput(.99, .10, .99, 1.0, 0))
        self.assertEqual((r.level, r.decision), ("low", "refuse"))

    def test_medium_boundary(self):
        r = calculate_confidence(EvidenceInput(.50, .50, .50, .50, 1))
        self.assertEqual(r.level, "medium")

    def test_below_medium(self):
        r = calculate_confidence(EvidenceInput(.40, .50, .40, .50, 1))
        self.assertEqual(r.level, "low")

    def test_high_requires_coverage(self):
        r = calculate_confidence(EvidenceInput(1.0, .79, 1.0, 1.0, 5))
        self.assertEqual(r.level, "medium")

    def test_high_requires_multiple_chunks(self):
        r = calculate_confidence(EvidenceInput(1.0, 1.0, 1.0, 1.0, 1))
        self.assertEqual(r.level, "medium")

    def test_config_loaded_from_json_matches_defaults(self):
        cfg = load_config_from_json()
        self.assertEqual(cfg.high_threshold, 0.75)
        self.assertEqual(cfg.medium_threshold, 0.50)
        self.assertEqual(cfg.retrieval_weight, 0.30)
        self.assertEqual(cfg.coverage_weight, 0.30)
        self.assertEqual(cfg.agreement_weight, 0.20)
        self.assertEqual(cfg.consistency_weight, 0.20)
        self.assertEqual(cfg.min_chunks_for_high, 2)
        self.assertEqual(cfg.min_coverage_for_high, 0.80)
        self.assertEqual(cfg.min_coverage_for_medium, 0.50)

    def test_calculate_confidence_uses_json_config_by_default(self):
        # Same inputs as test_high, but relying on the JSON-loaded config
        # (no explicit ConfidenceConfig passed) to prove it's wired in.
        r = calculate_confidence(EvidenceInput(.90, .95, .90, 1.0, 3))
        self.assertEqual((r.level, r.decision), ("high", "answer"))


if __name__ == "__main__":
    unittest.main()
