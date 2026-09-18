"""Preview an OpenCV capture device without running OCR."""

import argparse
import sys

import cv2


OCR_FRAME_SIZE = (1672, 941)  # Width, height expected by scoreboard.crop_scoreboard.
CAPTURE_SIZE = (1920, 1080)
CAPTURE_FPS = 60


def normalize_frame(frame):
    """Return a frame sized for the existing fixed scoreboard crops.

    Call this before crop_scoreboard(frame), then pass its crops to
    extract_game_state(). The preview itself uses the original capture frame.
    """
    if frame is None or getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Expected a nonempty BGR frame")
    if frame.shape[:2] == (OCR_FRAME_SIZE[1], OCR_FRAME_SIZE[0]):
        return frame.copy()
    return cv2.resize(frame, OCR_FRAME_SIZE, interpolation=cv2.INTER_AREA)


def preview_capture(device_index=0):
    """Show live frames until Q is pressed. Return 0 on success, 1 on failure."""
    if device_index < 0:
        raise ValueError("Device index must be nonnegative")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = None
    try:
        capture = cv2.VideoCapture(device_index, backend)
        if not capture.isOpened():
            print(f"Could not open capture device {device_index}.", file=sys.stderr)
            return 1
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        capture.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        reported = False
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                print("Capture device stopped providing frames.", file=sys.stderr)
                return 1
            if not reported:
                height, width = frame.shape[:2]
                print(f"Actual capture: {width} x {height} at "
                      f"{capture.get(cv2.CAP_PROP_FPS):g} FPS (device reported)")
                print("Press Q in the video window to exit.")
                reported = True
            cv2.imshow("NBA 2K Live Capture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return 0
    except cv2.error as error:
        print(f"Capture error: {error}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview an NBA 2K capture card")
    parser.add_argument("--device", type=int, default=0, help="OpenCV device index (default: 0)")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    try:
        return preview_capture(args.device)
    except KeyboardInterrupt:
        print("Capture stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
