"""Compare OCR variants: python src/vision/ocr_test.py."""

if __package__:
    from .ocr import IMAGE_DIR, read_text, extract_game_state
else:
    from ocr import IMAGE_DIR, read_text, extract_game_state


def main():
    # Expected values are for checking this fixture only, never OCR inputs.
    examples = [
        ("team_1_score", "18", "0123456789"),
        ("team_2_score", "21", "0123456789"),
        ("time_remaining", "3:30", "0123456789:."),
        ("quarter", "1st", "0123456789stndrhOTot"),
    ]
    variants = [
        ("original", {}),
        ("grayscale + 3x resize", {"preprocess": True}),
        ("grayscale + 3x resize + contrast + threshold",
         {"preprocess": True, "contrast": 1.5, "threshold": True}),
    ]
    for name, expected, allowlist in examples:
        image_path = IMAGE_DIR / f"test_image_{name}.png"
        print(f"\nImage: {image_path}\nExpected: {expected!r}", flush=True)
        for label, options in variants:
            results = read_text(image_path, allowlist=allowlist, **options)
            print(f"  Variant: {label}")
            print(f"  Detections: {len(results)}")
            for result in results:
                print(f"    Text: {result['text']!r}")
                print(f"    Confidence: {result['confidence']:.4f}")
                print(f"    Bounding box: {result['bounding_box']}")
            detected_text = " ".join(result["text"] for result in results)
            print(f"  Exact match: {detected_text == expected}", flush=True)

    crops = {name: IMAGE_DIR / f"test_image_{name}.png" for name, _, _ in examples}
    state = extract_game_state(crops)
    expected_state = {
        "team_1_score": 18, "team_2_score": 21,
        "quarter": "1st", "time_remaining": "3:30",
    }
    print(f"\nValidated game state: {state}")
    print(f"Matches test screenshot: {state == expected_state}")
    assert state == expected_state, "OCR did not match the test screenshot"


if __name__ == "__main__":
    main()
