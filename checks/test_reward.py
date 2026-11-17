"""Reference reward boundaries; not model quality or sandbox integration tests."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ksat_tool_rl_v2"))
from reward_reference import Outcome, reward

class RewardTests(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(reward(Outcome(correct=True)), 1.0)
    def test_incorrect(self):
        self.assertEqual(reward(Outcome(correct=False)), 0.0)
    def test_no_final(self):
        self.assertEqual(reward(Outcome(correct=True, complete=False)), -1.0)
    def test_invalid_protocol(self):
        self.assertEqual(reward(Outcome(correct=True, final_protocol_valid=False)), -1.0)
    def test_forbidden(self):
        self.assertEqual(reward(Outcome(correct=True, forbidden_action=True)), -1.0)
    def test_infra_not_wrong(self):
        self.assertIsNone(reward(Outcome(correct=False, infrastructure_error=True)))
    def test_no_bonus_in_initial_phase(self):
        self.assertEqual(reward(Outcome(correct=False, verified_evidence_score=1.0)), 0.0)
    def test_correct_floor(self):
        self.assertAlmostEqual(reward(Outcome(correct=True, normalized_runtime_cost=1.0), True), 0.95)
    def test_wrong_cannot_get_evidence_bonus(self):
        self.assertEqual(reward(Outcome(correct=False, verified_evidence_score=1.0), True), 0.0)
    def test_correct_ceiling(self):
        self.assertAlmostEqual(reward(Outcome(correct=True, verified_evidence_score=1.0), True), 1.1)
    def test_invalid_cost(self):
        for value in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reward(Outcome(correct=True, normalized_runtime_cost=value), True)
    def test_invalid_evidence(self):
        for value in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reward(Outcome(correct=True, verified_evidence_score=value), True)

if __name__ == "__main__":
    unittest.main()
