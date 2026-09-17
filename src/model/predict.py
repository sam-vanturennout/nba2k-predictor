"""Predict the home win probability from a validated live GameState."""

import joblib
import pandas as pd

from src.game_state import GameState
from src.model.train import FEATURE_COLUMNS


def predict_home_win_probability(state, bundle):
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if tuple(bundle["features"]) != FEATURE_COLUMNS:
        raise ValueError("Model features do not match this predictor")
    quarter_number = 5 if state.quarter == "OT" else int(state.quarter[:-2])
    features = pd.DataFrame([{
        "score_difference": state.score_difference,
        "quarter_number": quarter_number,
        "time_remaining_seconds": state.time_remaining_seconds,
    }], columns=list(FEATURE_COLUMNS))
    return float(bundle["model"].predict_proba(features)[0, 1])


def load_model(path):
    return joblib.load(path)
