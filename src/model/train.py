"""Train a game-grouped logistic regression baseline on real observations."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ("score_difference", "quarter_number", "time_remaining_seconds")
REQUIRED_COLUMNS = {"game_id", "data_type", "game_status", "home_wins",
                    "home_score", "away_score", "quarter", "time_remaining_seconds",
                    "home_final_score", "away_final_score"}
MIN_REAL_GAMES = 20


def prepare_observations(data):
    """Keep completed real games and construct features available at observation time."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    rows = data.loc[(data["data_type"] == "real") & (data["game_status"] == "completed")].copy()
    if rows.empty:
        raise ValueError("No completed real-game observations")
    if rows["game_id"].isna().any() or rows["game_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Completed real games need game_id")
    labels = pd.to_numeric(rows["home_wins"], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("home_wins must be 0 or 1 for every completed real observation")
    rows["home_wins"] = labels.astype(int)
    if rows.groupby("game_id")["home_wins"].nunique().gt(1).any():
        raise ValueError("Conflicting outcome labels within a game")
    final_scores = rows[["home_final_score", "away_final_score"]].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(final_scores.to_numpy(dtype=float)).all()
            or (final_scores < 0).any().any()
            or (final_scores % 1 != 0).any().any()
            or (final_scores["home_final_score"] == final_scores["away_final_score"]).any()):
        raise ValueError("Completed games need valid, non-tied final scores")
    if (rows["home_wins"] != (final_scores["home_final_score"] > final_scores["away_final_score"]).astype(int)).any():
        raise ValueError("home_wins disagrees with final scores")
    if final_scores.assign(game_id=rows["game_id"]).groupby("game_id").nunique().gt(1).any().any():
        raise ValueError("Conflicting final scores within a game")
    scores = rows[["home_score", "away_score", "time_remaining_seconds"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=float)).all() or (scores < 0).any().any():
        raise ValueError("Scores and time must be finite nonnegative numbers")
    if ((scores["home_score"] % 1 != 0) | (scores["away_score"] % 1 != 0)).any():
        raise ValueError("Scores must be integers")
    from src.vision.ocr import parse_quarter

    normalized = rows["quarter"].map(parse_quarter)
    if normalized.isna().any():
        raise ValueError("Invalid quarter in completed real observation")
    rows["score_difference"] = scores["home_score"] - scores["away_score"]
    rows["quarter_number"] = normalized.map(lambda q: 5 if q == "OT" else int(q[:-2]))
    rows["time_remaining_seconds"] = scores["time_remaining_seconds"]
    return rows


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y_true, predictions)),
            "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, probabilities))}


def train_model(data, *, min_games=MIN_REAL_GAMES, random_state=42):
    rows = prepare_observations(data)
    game_labels = rows.groupby("game_id")["home_wins"].first()
    if len(game_labels) < min_games:
        raise ValueError(f"Need at least {min_games} completed real games; found {len(game_labels)}")
    if game_labels.value_counts().min() < 2 or game_labels.nunique() != 2:
        raise ValueError("Need at least two home wins and two home losses")
    train_ids, test_ids = train_test_split(
        game_labels.index.to_numpy(), test_size=0.25, random_state=random_state,
        stratify=game_labels.to_numpy(),
    )
    training = rows[rows["game_id"].isin(train_ids)]
    testing = rows[rows["game_id"].isin(test_ids)]
    if training["home_wins"].nunique() != 2 or testing["home_wins"].nunique() != 2:
        raise ValueError("Train and test games must each include both outcomes")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(training[list(FEATURE_COLUMNS)], training["home_wins"])
    probabilities = model.predict_proba(testing[list(FEATURE_COLUMNS)])[:, 1]
    prevalence = float(training.groupby("game_id")["home_wins"].first().mean())
    baseline = np.full(len(testing), prevalence)
    metrics = {"model": evaluate(testing["home_wins"], probabilities),
               "baseline": evaluate(testing["home_wins"], baseline),
               "train_games": len(train_ids), "test_games": len(test_ids),
               "train_observations": len(training), "test_observations": len(testing)}
    return {"model": model, "features": FEATURE_COLUMNS}, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train on completed real NBA 2K observations")
    parser.add_argument("--input", default="data/processed/observations.csv")
    parser.add_argument("--output", default="models/win_probability.joblib")
    args = parser.parse_args(argv)
    try:
        bundle, metrics = train_model(pd.read_csv(args.input))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Training skipped: {error}\n")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Saved {output}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
