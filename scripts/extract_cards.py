"""Card extraction utility for Shotgun King reinforcement learning.

This script parses the window screenshot (data/screenshot.png) in 1280x720 resolution,
crops the 4 distinct card regions (top-left, top-right, bottom-left, bottom-right),
and saves them individually into the assets/cards/ directory as database templates.
"""

import os

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def extract_game_cards(image_path="data/screenshot.png", output_dir=".temp"):
    """Crops the four card templates from the screenshot and saves them.

    Args:
        image_path: Path to the 1280x720 screenshot.
        output_dir: Target directory to save cropped cards.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    if not os.path.exists(image_path):
        print(f"Error: Source screenshot {image_path} does not exist. Please run capture_screen first.")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Defined precise coordinates based on symmetrical offsets around 1280x720 center
    card_regions = {
        "card_top_left.png": {"x_start": 540, "x_end": 624, "y_start": 216, "y_end": 332},
        "card_top_right.png": {"x_start": 651, "x_end": 735, "y_start": 216, "y_end": 332},
        "card_bottom_left.png": {"x_start": 540, "x_end": 624, "y_start": 416, "y_end": 532},
        "card_bottom_right.png": {"x_start": 651, "x_end": 735, "y_start": 416, "y_end": 532},
    }

    if cv2 is not None and np is not None:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print("Error: OpenCV failed to read image.")
                return False

            h, w = img.shape[:2]
            if w != 1280 or h != 720:
                # Resize dynamically if resolution slightly differs to enforce standard coordinate matching
                img = cv2.resize(img, (1280, 720))

            for name, coords in card_regions.items():
                x1, x2 = coords["x_start"], coords["x_end"]
                y1, y2 = coords["y_start"], coords["y_end"]
                cropped = img[y1:y2, x1:x2]
                out_path = os.path.join(output_dir, name)
                cv2.imwrite(out_path, cropped)
                print(f"Successfully extracted and saved: {out_path} (size: {cropped.shape[1]}x{cropped.shape[0]})")
            return True
        except Exception as e:
            print(f"Exception during OpenCV card extraction: {e}")
            return False
    else:
        try:
            from PIL import Image
            img = Image.open(image_path)
            if img.size != (1280, 720):
                img = img.resize((1280, 720))

            for name, coords in card_regions.items():
                x1, x2 = coords["x_start"], coords["x_end"]
                y1, y2 = coords["y_start"], coords["y_end"]
                cropped = img.crop((x1, y1, x2, y2))
                out_path = os.path.join(output_dir, name)
                cropped.save(out_path)
                print(f"Successfully extracted and saved (PIL): {out_path} (size: {cropped.size[0]}x{cropped.size[1]})")
            return True
        except Exception as e:
            print(f"Exception during PIL card extraction: {e}")
            return False


if __name__ == "__main__":
    print("Starting card template database extraction utility...")
    extract_game_cards()
