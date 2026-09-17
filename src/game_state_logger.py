"""Append validated game observations to a JSON Lines file."""

import json
from pathlib import Path

from .game_state import GameState


def log_game_state(state: GameState, path):
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(state.to_dict()) + "\n")
