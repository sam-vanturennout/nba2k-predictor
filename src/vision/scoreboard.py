"""Fixed scoreboard regions for the current 1672 x 941 NBA 2K layout."""

# Coordinates are (left, top, right, bottom) relative to the full frame.
CROP_BOXES = {
    "scoreboard": (20, 864, 516, 920),
    "quarter": (714, 873, 769, 914),
    "time_remaining": (539, 873, 628, 914),
    "team_1_score": (113, 870, 182, 906),
    "team_2_score": (352, 869, 420, 906),
}


def crop_scoreboard(image):
    """Return independent crops from an OpenCV frame in the supported layout."""
    if image is None or image.shape[:2] != (941, 1672):
        raise ValueError("Crop coordinates require a 1672 x 941 screenshot.")
    return {
        name: image[top:bottom, left:right].copy()
        for name, (left, top, right, bottom) in CROP_BOXES.items()
    }
