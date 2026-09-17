"""GameState validation and OCR integration tests."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from src.game_state import GameState
from src.vision.ocr import extract_game_state


class GameStateTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(game_id="game-1", timestamp=datetime(2026, 9, 17, tzinfo=timezone.utc),
                           home_score=21, away_score=18, quarter="2nd", time_remaining_seconds=210)

    def test_difference_and_dictionary(self):
        state = GameState(**self.values)
        self.assertEqual(state.score_difference, 3)
        self.assertEqual(state.to_dict(), {**self.values, "timestamp": "2026-09-17T00:00:00+00:00",
                                           "score_difference": 3})

    def test_invalid_values(self):
        for changes in ({"game_id": ""}, {"timestamp": None}, {"home_score": -1},
                        {"away_score": True}, {"quarter": "5th"},
                        {"time_remaining_seconds": None}, {"time_remaining_seconds": float("nan")},
                        {"time_remaining_seconds": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GameState(**{**self.values, **changes})

    @patch("src.vision.ocr.read_game_clock", return_value="3:30")
    @patch("src.vision.ocr.read_quarter", return_value="2nd")
    @patch("src.vision.ocr.read_score", side_effect=[18, 21])
    def test_extract_object_maps_home_explicitly(self, read_score, read_quarter, read_clock):
        crops = dict.fromkeys(("team_1_score", "team_2_score", "quarter", "time_remaining"), "unused")
        state = extract_game_state(crops, as_dataclass=True, game_id="game-1",
                                   timestamp=self.values["timestamp"], home_team=2)
        self.assertIsInstance(state, GameState)
        self.assertEqual((state.home_score, state.away_score, state.score_difference), (21, 18, 3))
        read_score.side_effect = [None, 21]
        with self.assertRaises(ValueError):
            extract_game_state(crops, as_dataclass=True, game_id="game-1",
                               timestamp=self.values["timestamp"], home_team=2)


if __name__ == "__main__":
    unittest.main()
