"""Screen capture and processing module for Shotgun King reinforcement learning.

This module provides basic capabilities to capture the game screen (either the
entire screen or a specific window by title), save the screenshots, and
perform initial image analysis using OpenCV.
"""

import os
import subprocess
import time

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import mss
except ImportError:
    mss = None

try:
    import pywinctl as pwc
except Exception:
    # Window discovery is optional during headless validation.
    pwc = None


def find_window_geometry(title="Shotgun King"):
    """Finds the bounding box coordinates of a window matching the given title.

    First try using pywinctl, and fallback to wmctrl if it fails.
    Returns accurate window coordinates even in Linux multi-monitor setups.

    Args:
        title: Part of or full title of the target window.

    Returns:
        A dictionary containing {'left': x, 'top': y, 'width': w, 'height': h}
        if the window is found, None otherwise.
    """
    # 1. Try pywinctl
    if pwc is not None:
        try:
            windows = pwc.getAllWindows()
            for win in windows:
                if win.title and title.lower() in win.title.lower():
                    try:
                        win.activate()
                    except Exception:
                        pass
                    rect = win.box
                    return {
                        "top": int(rect.top),
                        "left": int(rect.left),
                        "width": int(rect.width),
                        "height": int(rect.height),
                    }
        except Exception:
            pass

    # 2. Fallback to wmctrl (Linux)
    try:
        result = subprocess.run(
            ["wmctrl", "-lG"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if title.lower() in line.lower():
                parts = line.split()
                x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                return {"left": x, "top": y, "width": w, "height": h}
    except Exception:
        pass

    print(f"Warning: No window containing '{title}' was found.")
    return None


def capture_screen(output_path="data/screenshot.png", window_title="Shotgun King"):
    """Captures the screen and saves it as an image file.

    It attempts to find the target window geometry first. If found, it captures
    only that window. Otherwise, it falls back to capturing the primary monitor.

    Args:
        output_path: The file path where the captured image will be saved.
        window_title: Target window title to crop the capture region.

    Returns:
        True if the capture was successful and saved, False otherwise.
    """
    if mss is None:
        print("Error: 'mss' library is not installed. Please install it using 'pip install mss'.")
        return False

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with mss.mss() as sct:
            # Acquire entire screen geometry to validate boundary containment
            screen = sct.monitors[0]
            monitor = find_window_geometry(window_title) if window_title else None

            if monitor is not None:
                # Check if the game window geometry exits the boundary of the virtual screen dynamically
                is_out = (
                    monitor["left"] < screen["left"] or
                    monitor["top"] < screen["top"] or
                    (monitor["left"] + monitor["width"]) > (screen["left"] + screen["width"]) or
                    (monitor["top"] + monitor["height"]) > (screen["top"] + screen["height"])
                )
                if is_out:
                    msg = f"Game window is out of monitor bounds! Window: {monitor}, Screen: {screen}. Please move the window inside the monitor."
                    print(f"Error: {msg}")
                    raise RuntimeError(msg)
            else:
                print("Falling back to primary monitor capture...")
                monitor = sct.monitors[1]

            screenshot = sct.grab(monitor)
            
            if cv2 is not None and np is not None:
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(output_path, img)
            else:
                from PIL import Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                img.save(output_path)
            
            # print(f"Screen successfully captured and saved to {output_path}")
            return True
    except Exception as e:
        print(f"Failed to capture screen: {e}")
        return False


def detect_grid_prototype(image_path, output_path="data/grid_detected.png"):
    """Performs prototype grid detection on the captured image using OpenCV.

    Args:
        image_path: Path to the source screenshot image.
        output_path: Path to save the processed image with detected features.

    Returns:
        True if processing was successful, False otherwise.
    """
    if cv2 is None or np is None:
        print("Error: 'opencv-python' or 'numpy' is not installed. Grid detection bypassed.")
        return False

    if not os.path.exists(image_path):
        print(f"Error: Source image {image_path} does not exist.")
        return False

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 50, 150)
        
        cv2.imwrite(output_path, edges)
        print(f"Grid detection prototype saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed in grid detection prototype: {e}")
        return False


if __name__ == "__main__":
    print("Starting targeted screen capture test...")
    # Attempt to capture the specific window "Shotgun King"
    if capture_screen(output_path="data/screenshot.png", window_title="Shotgun King"):
        print("Attempting grid detection on the captured window screenshot...")
        detect_grid_prototype("data/screenshot.png", "data/grid_detected.png")
