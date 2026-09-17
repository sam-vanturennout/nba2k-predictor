from pathlib import Path

import cv2

if __package__:
    from .scoreboard import crop_scoreboard
else:
    from scoreboard import crop_scoreboard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = PROJECT_ROOT / "data/raw/images/test_image.png"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/images"

def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    crops = crop_scoreboard(image)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, crop in crops.items():
        output_path = OUTPUT_DIR / f"test_image_{name}.png"
        if not cv2.imwrite(str(output_path), crop):
            raise OSError(f"Could not save crop: {output_path}")
        print(f"Saved {name}: {output_path}")


if __name__ == "__main__":
    main()
