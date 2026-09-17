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
    text = text.strip()
    if re.fullmatch(r"[0-9]{1,2}\.[0-5][0-9]", text):
        text = text.replace(".", ":")
    if re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]", text):
        return text
    if re.fullmatch(r"[0-9]{1,2}\.[0-9]", text) and float(text) < 60:
        return text
    return None


def parse_quarter(text):
    """Accept regulation ordinals or OT, ignoring OCR capitalization."""
    text = text.strip().lower()
    if re.fullmatch(r"(?:1st|2nd|3rd|4th)", text):
        return text
    return "OT" if text == "ot" else None


def parse_score(text):
    """Accept a whole, nonnegative score of up to three digits."""
    text = text.strip()
    return int(text) if re.fullmatch(r"[0-9]{1,3}", text) else None


def _read_field(image_path, parser, **options):
    results = read_text(image_path, **options)
    # A tight crop should contain one field. Reject missing/ambiguous detections
    # rather than concatenating unrelated boxes into an apparently valid value.
    if len(results) != 1:
        return None
    return parser(results[0]["text"])


def read_game_clock(image_path):
    """Return a validated clock string, or None when recognition is invalid."""
    return _read_field(image_path, parse_game_clock, preprocess=True,
                       allowlist="0123456789:.")


def read_quarter(image_path):
    """Return a regulation ordinal or OT, or None on invalid recognition."""
    return _read_field(image_path, parse_quarter, preprocess=True,
                       contrast=1.5, threshold=True,
                       allowlist="0123456789stndrhOTot")


def read_score(image_path):
    """Return an integer score, or None on invalid recognition."""
    return _read_field(image_path, parse_score, preprocess=True,
                       allowlist="0123456789")


def extract_game_state(crops):
    """Read named crop paths or OpenCV arrays; failed fields remain None.

    Pass the dictionary returned by scoreboard.crop_scoreboard(frame), or a
    mapping of the same field names to saved crop paths. File/shape errors
    propagate to the caller; they are distinct from unsuccessful recognition.
    """
    return {
        "team_1_score": read_score(crops["team_1_score"]),
        "team_2_score": read_score(crops["team_2_score"]),
        "quarter": read_quarter(crops["quarter"]),
        "time_remaining": read_game_clock(crops["time_remaining"]),
    }
