"""Live video capture; OCR and persistence are supplied as callables."""

import argparse
from datetime import datetime, timezone
from functools import partial
import time

import cv2

from src.game_state_logger import log_game_state
from src.vision.ocr import extract_game_state
from src.vision.scoreboard import crop_scoreboard


def _previous_ocr_state(state, home_team):
    if state is None:
        return None
    return {
        "team_1_score": state.home_score if home_team == 1 else state.away_score,
        "team_2_score": state.away_score if home_team == 1 else state.home_score,
        "quarter": state.quarter,
        "time_remaining_seconds": state.time_remaining_seconds,
    }


def observe_frame(frame, previous_state, *, game_id, home_team,
                  regulation_quarters=4, period_seconds=None):
    """Run a frame through the existing crop, OCR, and GameState validation."""
    return extract_game_state(
        crop_scoreboard(frame),
        previous_state=_previous_ocr_state(previous_state, home_team),
        regulation_quarters=regulation_quarters,
        period_seconds=period_seconds,
        as_dataclass=True,
        game_id=game_id,
        timestamp=datetime.now(timezone.utc),
        home_team=home_team,
    )


def _observation_key(state):
    return (state.game_id, state.home_score, state.away_score,
            state.quarter, state.time_remaining_seconds)


def run_capture(capture, observe, save, *, interval=1.0, stop_event=None,
                max_read_failures=3, clock=time.monotonic, sleep=time.sleep):
    """Sample until stopped or disconnected; return the number of saved states.

    ``observe(frame, previous_state)`` returns a validated GameState.
    The caller owns and releases ``capture``.
    """
    if interval <= 0 or max_read_failures < 1:
        raise ValueError("interval must be positive and max_read_failures at least one")
    if not capture.isOpened():
        raise RuntimeError("Capture device is not connected")
    previous = None
    failures = saved = 0
    while stop_event is None or not stop_event.is_set():
        started = clock()
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            failures += 1
            if failures >= max_read_failures:
                raise RuntimeError("Capture device disconnected or frames are unreadable")
        else:
            failures = 0
            try:
                state = observe(frame, previous)
            except ValueError:
                # Invalid crops, incomplete OCR, or impossible transitions.
                pass
            else:
                if previous is None or _observation_key(state) != _observation_key(previous):
                    save(state)
                    previous = state
                    saved += 1
        delay = interval - (clock() - started)
        if delay > 0:
            if stop_event is not None:
                stop_event.wait(delay)
            else:
                sleep(delay)
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture validated NBA 2K scoreboard observations")
    parser.add_argument("--device", type=int, default=0, help="OpenCV capture device index")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--home-team", type=int, choices=(1, 2), required=True,
                        help="Scoreboard team number that is home")
    parser.add_argument("--output", default="data/processed/game_states.jsonl")
    parser.add_argument("--regulation-quarters", type=int, default=4)
    parser.add_argument("--period-seconds", type=float)
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.regulation_quarters < 1 or (args.period_seconds is not None and args.period_seconds <= 0):
        parser.error("interval, regulation-quarters, and period-seconds must be positive")
    capture = cv2.VideoCapture(args.device)
    try:
        observe = partial(observe_frame, game_id=args.game_id, home_team=args.home_team,
                          regulation_quarters=args.regulation_quarters,
                          period_seconds=args.period_seconds)
        save = partial(log_game_state, path=args.output)
        try:
            count = run_capture(capture, observe, save, interval=args.interval)
        except KeyboardInterrupt:
            print("Capture stopped")
            return 0
        print(f"Saved {count} observations")
        return 0
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
