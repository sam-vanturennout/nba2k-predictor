"""Reusable OCR for cropped NBA 2K scoreboard fields."""

from functools import lru_cache
from pathlib import Path
import re

import cv2
import easyocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data/processed/images"
MODEL_DIR = PROJECT_ROOT / "models/easyocr"


@lru_cache(maxsize=1)
def get_reader():
    """Load the English OCR models once, on first use, using the CPU."""
    # EasyOCR downloads missing weights on first run; later runs reuse them.
    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(MODEL_DIR),
        user_network_directory=str(MODEL_DIR / "user_network"),
    )


def preprocess_image(image, *, scale=3, contrast=1.0, threshold=False):
    """Return a grayscale/upscaled copy, optionally adjusting contrast/binarizing."""
    if scale <= 0 or contrast <= 0:
        raise ValueError("scale and contrast must be positive")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=contrast)
    if threshold:
        # Otsu chooses a threshold from this crop's brightness distribution.
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray


def read_text(image_path, *, preprocess=False, scale=3, contrast=1.0,
              threshold=False, allowlist=None):
    """Return dictionaries containing text, confidence, and bounding_box.

    Bounding boxes refer to the image passed to OCR (upscaled when preprocessing).
    An empty list means no text was detected. No expected answer is substituted.
    """
    # Accept saved crops or in-memory crops from a captured frame.
    image = cv2.imread(str(image_path)) if isinstance(image_path, (str, Path)) else image_path
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if preprocess:
        image = preprocess_image(image, scale=scale, contrast=contrast, threshold=threshold)

    # detail=1 preserves each detection's bounding box and confidence score.
    detections = get_reader().readtext(
        image, detail=1, paragraph=False, allowlist=allowlist, workers=0,
    )
    return [
        {
            "text": text,
            "confidence": float(confidence),
            "bounding_box": [[float(x), float(y)] for x, y in box],
        }
        for box, text, confidence in detections
    ]


def parse_game_clock(text):
    """Validate M:SS or seconds.tenths; normalize a misread colon in M.SS.

    This layout shows one decimal place for fractional seconds. Two digits
    after a dot are treated as the common colon-to-dot OCR error, not hundredths.
    No missing punctuation or digits are invented.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if re.fullmatch(r"[0-9]{1,2}\.[0-5][0-9]", text):
        text = text.replace(".", ":")
    if re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]", text):
        return text
    if re.fullmatch(r"[0-9]{1,2}\.[0-9]", text) and float(text) < 60:
        return text
    return None


def clock_to_seconds(text):
    """Convert a validated clock to seconds, or return None."""
    clock = parse_game_clock(text)
    if clock is None:
        return None
    if ":" in clock:
        minutes, seconds = map(int, clock.split(":"))
        return minutes * 60 + seconds
    return float(clock)


def parse_quarter(text, *, regulation_quarters=4):
    """Accept regulation ordinals or OT, ignoring OCR capitalization."""
    if not isinstance(text, str) or not isinstance(regulation_quarters, int) or regulation_quarters < 1:
        return None
    text = text.strip().lower()
    if text == "ot":
        return "OT"
    match = re.fullmatch(r"([1-9][0-9]*)(st|nd|rd|th)", text)
    if match is None:
        return None
    number = int(match.group(1))
    suffix = "th" if number % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return text if number <= regulation_quarters and match.group(2) == suffix else None


def parse_score(text):
    """Accept a whole, nonnegative score of up to three digits."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    return int(text) if re.fullmatch(r"[0-9]{1,3}", text) else None


def _read_field(image_path, parser, **options):
    results = read_text(image_path, **options)
    # A tight crop should contain one field. Reject missing/ambiguous detections
    # rather than concatenating unrelated boxes into an apparently valid value.
    if len(results) != 1:
        return None
    return parser(results[0].get("text")) if isinstance(results[0], dict) else None


def read_game_clock(image_path):
    """Return a validated clock string, or None when recognition is invalid."""
    return _read_field(image_path, parse_game_clock, preprocess=True,
                       allowlist="0123456789:.")


def read_quarter(image_path, *, regulation_quarters=4):
    """Return a regulation ordinal or OT, or None on invalid recognition."""
    return _read_field(image_path, lambda text: parse_quarter(text, regulation_quarters=regulation_quarters), preprocess=True,
                       contrast=1.5, threshold=True,
                       allowlist="0123456789stndrhOTot")


def read_score(image_path):
    """Return an integer score, or None on invalid recognition."""
    return _read_field(image_path, parse_score, preprocess=True,
                       allowlist="0123456789")


def validate_game_transition(previous_state, state):
    """Reject score decreases and backward periods within the same game."""
    if previous_state is None:
        return True
    for key in ("team_1_score", "team_2_score"):
        old, new = previous_state.get(key), state.get(key)
        if old is not None and new is not None and new < old:
            return False
    old_quarter, new_quarter = previous_state.get("quarter"), state.get("quarter")
    if old_quarter and new_quarter:
        old_period = float("inf") if old_quarter == "OT" else int(old_quarter[:-2])
        new_period = float("inf") if new_quarter == "OT" else int(new_quarter[:-2])
        if new_period < old_period:
            return False
        old_time = previous_state.get("time_remaining_seconds")
        new_time = state.get("time_remaining_seconds")
        if new_period == old_period and new_quarter != "OT" and old_time is not None and new_time is not None and new_time > old_time:
            return False
    return True


def extract_game_state(crops, *, previous_state=None, regulation_quarters=4,
                       period_seconds=None, as_dataclass=False, game_id=None,
                       timestamp=None, home_team=None):
    """Read named crop paths or OpenCV arrays; failed fields remain None.

    Pass the dictionary returned by scoreboard.crop_scoreboard(frame), or a
    mapping of the same field names to saved crop paths. File/shape errors
    propagate to the caller; they are distinct from unsuccessful recognition.
    """
    state = {
        "team_1_score": read_score(crops["team_1_score"]),
        "team_2_score": read_score(crops["team_2_score"]),
        "quarter": read_quarter(crops["quarter"], regulation_quarters=regulation_quarters),
        "time_remaining": read_game_clock(crops["time_remaining"]),
    }
    state["time_remaining_seconds"] = clock_to_seconds(state["time_remaining"])
    if period_seconds is not None:
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        if state["time_remaining_seconds"] is not None and state["time_remaining_seconds"] > period_seconds:
            state["time_remaining"] = None
            state["time_remaining_seconds"] = None
    if not validate_game_transition(previous_state, state):
        raise ValueError("Impossible game-state transition")
    if as_dataclass:
        if home_team not in (1, 2) or isinstance(home_team, bool):
            raise ValueError("home_team must be 1 or 2")
        from src.game_state import GameState

        return GameState(
            game_id=game_id,
            timestamp=timestamp,
            home_score=state[f"team_{home_team}_score"],
            away_score=state[f"team_{3 - home_team}_score"],
            quarter=state["quarter"],
            time_remaining_seconds=state["time_remaining_seconds"],
            regulation_quarters=regulation_quarters,
        )
    return state
