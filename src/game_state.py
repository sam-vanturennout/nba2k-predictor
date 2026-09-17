"""Validated game observation shared by collection and prediction code."""

from dataclasses import InitVar, asdict, dataclass, field
from datetime import datetime
from math import isfinite


@dataclass(frozen=True)
class GameState:
    game_id: str
    timestamp: datetime
    home_score: int
    away_score: int
    quarter: str
    time_remaining_seconds: float
    score_difference: int = field(init=False)
    regulation_quarters: InitVar[int] = 4

    def __post_init__(self, regulation_quarters):
        if not isinstance(self.game_id, str) or not self.game_id.strip():
            raise ValueError("game_id must be a nonempty string")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime")
        for name in ("home_score", "away_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        # Use the same quarter rules as OCR, including the current four quarters.
        from .vision.ocr import parse_quarter

        if parse_quarter(self.quarter, regulation_quarters=regulation_quarters) != self.quarter:
            raise ValueError("quarter must be a valid regulation quarter or OT")
        seconds = self.time_remaining_seconds
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
            raise ValueError("time_remaining_seconds must be a finite nonnegative number")
        object.__setattr__(self, "score_difference", self.home_score - self.away_score)

    def to_dict(self):
        """Return a JSON friendly record with an ISO 8601 timestamp."""
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}
