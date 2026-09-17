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

Scores and clock use grayscale with 3x resizing. Quarter also uses contrast 1.5
and Otsu thresholding. Invalid or ambiguous OCR fields return `None`; unreadable
files raise an error. Clock validation supports M:SS and seconds with one decimal
place. It assumes fractional clocks show tenths: a dot followed by two digits
in the seconds range is normalized to a colon. Hundredths displays would require
revisiting this assumption. Accuracy has only been checked on this screenshot;
crop coordinates are specific to its layout. No win-prediction model is included.
