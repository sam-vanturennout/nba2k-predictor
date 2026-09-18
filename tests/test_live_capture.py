"""Hardware-free tests for the live video preview."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.vision.live_capture import normalize_frame, preview_capture
from src.vision.scoreboard import crop_scoreboard


class LiveCaptureTests(unittest.TestCase):
    def test_normalize_frame_preserves_crop_dimensions(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        normalized = normalize_frame(frame)
        self.assertEqual(normalized.shape, (941, 1672, 3))
        self.assertEqual(crop_scoreboard(normalized)["team_1_score"].shape, (36, 69, 3))

    def test_normalize_frame_copies_existing_size_and_rejects_invalid(self):
        frame = np.zeros((941, 1672, 3), dtype=np.uint8)
        normalized = normalize_frame(frame)
        self.assertIsNot(normalized, frame)
        for invalid in (None, np.empty((0, 0, 3)), np.zeros((10, 10))):
            with self.subTest(invalid=repr(invalid)[:30]), self.assertRaises(ValueError):
                normalize_frame(invalid)

    @patch("src.vision.live_capture.sys.platform", "win32")
    @patch("src.vision.live_capture.cv2.destroyAllWindows")
    @patch("src.vision.live_capture.cv2.waitKey", return_value=ord("q"))
    @patch("src.vision.live_capture.cv2.imshow")
    @patch("src.vision.live_capture.cv2.VideoCapture")
    def test_preview_configures_device_and_releases(self, video_capture, imshow, wait_key, destroy):
        capture = video_capture.return_value
        capture.isOpened.return_value = True
        capture.read.return_value = True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture.get.return_value = 60
        self.assertEqual(preview_capture(2), 0)
        video_capture.assert_called_once_with(2, cv2.CAP_DSHOW)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        capture.set.assert_any_call(cv2.CAP_PROP_FPS, 60)
        capture.release.assert_called_once()
        destroy.assert_called_once()

    @patch("src.vision.live_capture.cv2.destroyAllWindows")
    @patch("src.vision.live_capture.cv2.VideoCapture")
    def test_open_and_read_failures_release_device(self, video_capture, destroy):
        capture = video_capture.return_value
        capture.isOpened.return_value = False
        self.assertEqual(preview_capture(), 1)
        capture.isOpened.return_value = True
        capture.read.return_value = False, None
        self.assertEqual(preview_capture(), 1)
        self.assertEqual(capture.release.call_count, 2)
        self.assertEqual(destroy.call_count, 2)


if __name__ == "__main__":
    unittest.main()
