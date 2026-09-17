"""Feature and prediction tests without fitting on fabricated outcomes."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock

import pandas as pd
import numpy as np

from src.game_state import GameState
from src.model.predict import predict_home_win_probability
from src.model.train import FEATURE_COLUMNS, prepare_observations, train_model


class ModelTests(unittest.TestCase):
    def test_features_use_only_current_observation(self):
        data = pd.DataFrame([{
            "game_id": "real-game", "data_type": "real", "game_status": "completed",
            "home_wins": 1, "home_score": 20, "away_score": 24,
            "quarter": "2nd", "time_remaining_seconds": 90,
            "home_final_score": 80, "away_final_score": 60,
        }])
        rows = prepare_observations(data)
        self.assertEqual(rows.iloc[0]["score_difference"], -4)
        self.assertEqual(rows.iloc[0]["quarter_number"], 2)
        self.assertEqual(tuple(FEATURE_COLUMNS), ("score_difference", "quarter_number", "time_remaining_seconds"))
        with self.assertRaisesRegex(ValueError, "at least 20"):
            train_model(data)

    def test_invalid_outcome_and_synthetic_only(self):
        row = dict(game_id="g1", data_type="real", game_status="completed", home_wins=2,
                   home_score=10, away_score=8, quarter="1st", time_remaining_seconds=120,
                   home_final_score=20, away_final_score=18)
        with self.assertRaisesRegex(ValueError, "home_wins"):
            prepare_observations(pd.DataFrame([row]))
        with self.assertRaisesRegex(ValueError, "No completed real"):
            prepare_observations(pd.DataFrame([{**row, "data_type": "synthetic"}]))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_observations(pd.DataFrame([{**row, "home_wins": 0}]))

    def test_prediction_uses_live_state_fields(self):
        state = GameState("g1", datetime(2026, 9, 17, tzinfo=timezone.utc), 30, 24, "3rd", 42)
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        probability = predict_home_win_probability(state, {"model": model, "features": FEATURE_COLUMNS})
        self.assertAlmostEqual(probability, 0.7)
        features = model.predict_proba.call_args.args[0].iloc[0]
        self.assertEqual(features.to_dict(), {"score_difference": 6, "quarter_number": 3,
                                             "time_remaining_seconds": 42})


if __name__ == "__main__":
    unittest.main()
