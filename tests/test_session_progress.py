import unittest

from utils.session_progress import StageProgressTracker


def passing_feedback(score):
    return {
        "passed": True,
        "metrics": {"final_score": score},
    }


class StageProgressTrackerTests(unittest.TestCase):
    def test_failure_resets_pass_hold(self):
        tracker = StageProgressTracker(["address", "takeaway"], pass_hold_sec=2.0)
        tracker.update(10.0, passing_feedback(90), stable=True)
        self.assertGreater(tracker.pass_progress(11.0), 0.0)

        tracker.update(11.1, {"passed": False}, stable=True)
        self.assertEqual(tracker.pass_progress(11.2), 0.0)

    def test_unstable_pose_resets_pass_hold(self):
        tracker = StageProgressTracker(["address", "takeaway"], pass_hold_sec=2.0)
        tracker.update(10.0, passing_feedback(90), stable=True)
        tracker.update(11.0, passing_feedback(90), stable=False)
        self.assertEqual(tracker.pass_progress(11.1), 0.0)

    def test_pass_hold_advances_and_saves_score(self):
        tracker = StageProgressTracker(["address", "takeaway"], pass_hold_sec=2.0)
        tracker.update(10.0, passing_feedback(88), stable=True)
        event = tracker.update(12.0, passing_feedback(88), stable=True)

        self.assertTrue(event["advanced"])
        self.assertEqual(tracker.current_stage_key, "takeaway")
        self.assertEqual(tracker.scores["address"], 88)

    def test_last_stage_completes_session(self):
        tracker = StageProgressTracker(["address"], pass_hold_sec=1.0)
        tracker.update(5.0, passing_feedback(93), stable=True)
        event = tracker.update(6.0, passing_feedback(93), stable=True)

        self.assertTrue(event["completed"])
        self.assertTrue(tracker.completed)
        self.assertEqual(tracker.summary()["average_score"], 93)

    def test_manual_stage_selection_clears_active_hold(self):
        tracker = StageProgressTracker(["address", "takeaway"], pass_hold_sec=2.0)
        tracker.update(10.0, passing_feedback(90), stable=True)
        tracker.select_stage(1)
        self.assertEqual(tracker.current_stage_key, "takeaway")
        self.assertEqual(tracker.pass_progress(11.0), 0.0)


if __name__ == "__main__":
    unittest.main()
