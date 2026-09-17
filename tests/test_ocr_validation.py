"""Validation tests run without loading OCR models."""

import unittest
from unittest.mock import patch

from src.vision.ocr import parse_game_clock, parse_quarter, parse_score, read_score


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

    def test_quarters(self):
        for text in ("1st", "2nd", "3rd", "4th", "OT"):
            self.assertEqual(parse_quarter(text), text)
        self.assertEqual(parse_quarter(" ot "), "OT")
        for text in ("", "5th", "1nd", "Tst", "1", "1st 2nd"):
            self.assertIsNone(parse_quarter(text))

    def test_scores(self):
        self.assertEqual(parse_score("0"), 0)
        self.assertEqual(parse_score("123"), 123)
        for text in ("", "-1", "18.0", "1 8", "1000", "abc"):
            self.assertIsNone(parse_score(text))

    @patch("src.vision.ocr.read_text")
    def test_missing_ambiguous_or_invalid_detection(self, read_text):
        for results in ([], [{"text": "1"}, {"text": "8"}], [{"text": "bad"}]):
            read_text.return_value = results
            self.assertIsNone(read_score("unused.png"))


if __name__ == "__main__":
    unittest.main()
