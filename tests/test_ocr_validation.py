"""Validation tests run without loading OCR models."""

import unittest
from unittest.mock import patch

from src.vision.ocr import (clock_to_seconds, extract_game_state, parse_game_clock,
                            parse_quarter, parse_score, read_score,
                            validate_game_transition)


class OCRValidationTests(unittest.TestCase):
    def test_clock_formats(self):
        for text in ("3:30", "0:42", "12.5", "0.0", "59.9"):
            with self.subTest(text=text):
                self.assertEqual(parse_game_clock(text), text)
        self.assertEqual(parse_game_clock("3.30"), "3:30")

    def test_invalid_clocks(self):
        for text in ("", "330", "3:60", "60.0", "3:3", "12.999", "abc", "-1.5"):
            with self.subTest(text=text):
                self.assertIsNone(parse_game_clock(text))

    def test_clock_seconds(self):
        self.assertEqual(clock_to_seconds("3.30"), 210)
        self.assertEqual(clock_to_seconds("12.5"), 12.5)
        self.assertIsNone(clock_to_seconds(None))
        self.assertIsNone(clock_to_seconds("3:60"))

    def test_quarters(self):
        for text in ("1st", "2nd", "3rd", "4th", "OT"):
            self.assertEqual(parse_quarter(text), text)
        self.assertEqual(parse_quarter(" ot "), "OT")
        for text in ("", "5th", "1nd", "Tst", "1", "1st 2nd"):
            self.assertIsNone(parse_quarter(text))
        self.assertEqual(parse_quarter("2nd", regulation_quarters=2), "2nd")
        self.assertIsNone(parse_quarter("3rd", regulation_quarters=2))
        self.assertIsNone(parse_quarter(None))

    def test_scores(self):
        self.assertEqual(parse_score("0"), 0)
        self.assertEqual(parse_score("123"), 123)
        for text in ("", "-1", "18.0", "1 8", "1000", "abc"):
            self.assertIsNone(parse_score(text))
        self.assertIsNone(parse_score(None))

    @patch("src.vision.ocr.read_text")
    def test_missing_ambiguous_or_invalid_detection(self, read_text):
        for results in ([], [{"text": "1"}, {"text": "8"}], [{"text": "bad"}]):
            read_text.return_value = results
            self.assertIsNone(read_score("unused.png"))
        read_text.return_value = [{}]
        self.assertIsNone(read_score("unused.png"))

    @patch("src.vision.ocr.read_game_clock")
    @patch("src.vision.ocr.read_quarter")
    @patch("src.vision.ocr.read_score")
    def test_extract_state_and_rules(self, read_score, read_quarter, read_clock):
        crops = dict.fromkeys(("team_1_score", "team_2_score", "quarter", "time_remaining"), "unused")
        read_score.side_effect = [10, 12]
        read_quarter.return_value = "2nd"
        read_clock.return_value = "3:30"
        state = extract_game_state(crops, regulation_quarters=2, period_seconds=300)
        self.assertEqual(state["time_remaining_seconds"], 210)
        self.assertEqual(state["team_1_score"], 10)
        read_score.side_effect = [None, None]
        read_clock.return_value = None
        missing = extract_game_state(crops)
        self.assertIsNone(missing["time_remaining_seconds"])
        self.assertIsNone(missing["team_1_score"])
        read_score.side_effect = [10, 12]
        read_clock.return_value = "6:00"
        self.assertIsNone(extract_game_state(crops, period_seconds=300)["time_remaining_seconds"])
        read_score.side_effect = [9, 12]
        read_clock.return_value = "3:00"
        with self.assertRaisesRegex(ValueError, "Impossible game-state transition"):
            extract_game_state(crops, previous_state=state)

    def test_impossible_transitions(self):
        previous = {"team_1_score": 10, "team_2_score": 12,
                    "quarter": "2nd", "time_remaining_seconds": 210}
        self.assertFalse(validate_game_transition(previous, {**previous, "team_1_score": 9}))
        self.assertFalse(validate_game_transition(previous, {**previous, "quarter": "1st"}))
        self.assertFalse(validate_game_transition(previous, {**previous, "time_remaining_seconds": 220}))
        self.assertTrue(validate_game_transition(previous, {**previous, "team_1_score": None}))
        self.assertTrue(validate_game_transition(previous, {**previous, "quarter": "3rd", "time_remaining_seconds": 300}))


if __name__ == "__main__":
    unittest.main()
