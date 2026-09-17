"""Live capture tests use fake frames and no physical device."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from src.game_state import GameState
from src.game_state_logger import log_game_state
from src.vision.capture import main, observe_frame, run_capture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((941, 1672, 3), dtype=np.uint8)
        self.first = GameState("game-1", datetime(2026, 9, 17, tzinfo=timezone.utc), 10, 12, "1st", 180)
        self.second = GameState("game-1", datetime(2026, 9, 17, 0, 0, 1, tzinfo=timezone.utc), 12, 12, "1st", 175)

    def test_save_changes_only_and_stop(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, self.frame)] * 3
        stop = Mock()
        stop.is_set.side_effect = [False, False, False, True]
        stop.wait.return_value = None
        observe = Mock(side_effect=[self.first, self.first, self.second])
        save = Mock()
        self.assertEqual(run_capture(capture, observe, save, stop_event=stop), 2)
        self.assertEqual(save.call_count, 2)
        self.assertIsNone(observe.call_args_list[0].args[1])
        self.assertEqual(observe.call_args_list[2].args[1], self.first)

    def test_invalid_frame_and_disconnection(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(False, None), (True, self.frame), (False, None), (False, None)]
        observe = Mock(side_effect=ValueError("unreadable OCR"))
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            run_capture(capture, observe, Mock(), max_read_failures=2, sleep=lambda _: None)
        self.assertEqual(observe.call_count, 1)
        capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            run_capture(capture, observe, Mock())

    @patch("src.vision.capture.extract_game_state")
    def test_frame_uses_existing_pipeline(self, extract):
        extract.return_value = self.second
        result = observe_frame(self.frame, self.first, game_id="game-1", home_team=2)
        self.assertEqual(result, self.second)
        self.assertEqual(extract.call_args.kwargs["previous_state"]["team_1_score"], 12)
        self.assertEqual(extract.call_args.kwargs["previous_state"]["team_2_score"], 10)
        self.assertEqual(extract.call_args.kwargs["home_team"], 2)

    def test_logger_writes_json_lines(self):
        stream = mock_open()
        path = Path("data/processed/observations.jsonl")
        with patch.object(Path, "open", stream), patch.object(Path, "mkdir"):
            log_game_state(self.first, path)
            log_game_state(self.second, path)
        rows = [json.loads(call.args[0]) for call in stream().write.call_args_list]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["score_difference"], -2)

    @patch("src.vision.capture.cv2.VideoCapture")
    def test_cli_releases_disconnected_device(self, video_capture):
        video_capture.return_value.isOpened.return_value = False
        with self.assertRaises(SystemExit) as result:
            main(["--device", "2", "--game-id", "game-1", "--home-team", "1"])
        self.assertEqual(result.exception.code, 1)
        video_capture.assert_called_once_with(2)
        video_capture.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
