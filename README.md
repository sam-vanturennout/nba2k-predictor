# nba2k-predictor
Uses image recognition and historical data to live predict current matchups

Current prototype extracts scoreboard fields from the supplied 1672 x 941 screenshot.
Run from the project root (EasyOCR weights download on first use):

```powershell
.venv/Scripts/python.exe src/vision/test_image.py
.venv/Scripts/python.exe src/vision/ocr_test.py
.venv/Scripts/python.exe -m unittest discover -s tests
```

`scoreboard.py` owns crop coordinates; `test_image.py` saves the five crops.
`ocr.py` caches the EasyOCR reader and exposes `read_game_clock`, `read_quarter`,
`read_score`, and `extract_game_state`. Field readers accept paths or OpenCV arrays.
`ocr_test.py` compares preprocessing variants and checks the fixture's game state.

To extract directly from a frame without saving intermediate files:

```python
import cv2
from src.vision.scoreboard import crop_scoreboard
from src.vision.ocr import extract_game_state

frame = cv2.imread("data/raw/images/test_image.png")
state = extract_game_state(crop_scoreboard(frame))
print(state)
```

For a validated record, pass the capture metadata and identify which scoreboard
score belongs to the home team. The default call above still returns a dictionary.
An incomplete OCR reading raises `ValueError` when a `GameState` is requested.

```python
from datetime import datetime, timezone

record = extract_game_state(
    crop_scoreboard(frame), as_dataclass=True, game_id="your-game-id",
    timestamp=datetime.now(timezone.utc), home_team=2,
)
saved_row = record.to_dict()
```

The collection pipeline can save `saved_row` with its ISO timestamp. The
prediction pipeline can consume `record.home_score`, `record.away_score`,
`record.score_difference`, `record.quarter`, and `record.time_remaining_seconds`
as validated inputs. `home_team` refers to `team_1_score` or `team_2_score` in
the OCR output; set it from the known matchup metadata.

To capture from an Xbox capture card, select its OpenCV device index and supply
the game ID and home team mapping:

```powershell
.venv/Scripts/python.exe -m src.vision.capture --device 0 --interval 1.0 --game-id game-001 --home-team 2 --period-seconds 300 --output data/processed/game_states.jsonl
```

Press Ctrl+C to stop and release the device. The command samples every second,
rejects incomplete or impossible readings, and appends changed observations as
JSON Lines records. `--period-seconds` should match your NBA 2K quarter length;
omit it if unknown. The current crop coordinates require 1672 x 941 frames.
After three unreadable frames the command exits with a connection error. A new
run starts a new in-memory duplicate and transition baseline.

Scores and clock use grayscale with 3x resizing. Quarter also uses contrast 1.5
and Otsu thresholding. Invalid or ambiguous OCR fields return `None`; unreadable
files raise an error. Clock validation supports M:SS and seconds with one decimal
place. It assumes fractional clocks show tenths: a dot followed by two digits
in the seconds range is normalized to a colon. Hundredths displays would require
revisiting this assumption. Accuracy has only been checked on this screenshot;
crop coordinates are specific to its layout.

## Win probability baseline

The current `data/processed/games.csv` contains 800 synthetic game summaries. It
does not contain live observations, a home-side mapping, or verified home-win
labels, so it is ineligible for training. Collect real observations and join
each to a verified final outcome before running:

```powershell
.venv/Scripts/python.exe -m src.model.train --input data/processed/observations.csv --output models/win_probability.joblib
```

The input CSV needs `game_id`, `data_type` (`real`), `game_status`
(`completed`), `home_wins` (0 or 1), `home_score`, `away_score`, `quarter`, and
`time_remaining_seconds`, `home_final_score`, and `away_final_score`.
Final scores verify the outcome label and are never model features. Each row
represents one observation during a game;
repeat the final `home_wins` label across that game's observations only after
the result is known. The trainer requires at least 20 distinct completed real
games and both outcomes. It splits by `game_id`, then reports accuracy, log
loss, and Brier score against a training-set home-win-rate baseline on held-out
games. Its features are only the observed score difference, quarter, and clock.
It does not use final scores or outcome fields as features. Player matchup is
excluded until real, observation-time matchup data is available. Successful
training saves a joblib bundle; `src.model.predict.predict_home_win_probability`
accepts that bundle and a live `GameState`.
