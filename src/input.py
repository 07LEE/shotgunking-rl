"""Input control and action simulation module for Shotgun King.

This module encapsulates keyboard and mouse simulation routines to control
the Shotgun King game window automatically.
"""

import os
import time

from shotgun_king_rl.runtime.io import IOStatus, InputResult

try:
    import pyautogui
    if pyautogui is not None:
        pyautogui.FAILSAFE = True
except Exception:
    # Keep pure imports usable in headless validation environments.
    pyautogui = None

from capture import find_window_geometry


def click_at(x, y, duration=0.1):
    """Simulates a mouse click at the given absolute screen coordinates.

    It performs a robust mouseDown-sleep-mouseUp sequence to ensure almost all
    game engines (like Unity, Godot) successfully register the mouse click.

    Args:
        x: Absolute X-coordinate on the screen.
        y: Absolute Y-coordinate on the screen.
        duration: Time in seconds taken to move the mouse pointer.

    Returns:
        A structured result describing success or failure.
    """
    if pyautogui is None:
        print("Error: 'pyautogui' is not installed.")
        return InputResult(IOStatus.UNAVAILABLE, "click", "pyautogui_unavailable", (x, y))

    mouse_down = False
    try:
        # Move mouse and perform an explicit click hold (mouseDown -> sleep -> mouseUp)
        pyautogui.moveTo(x, y, duration=duration)
        pyautogui.mouseDown()
        mouse_down = True
        time.sleep(0.05)
        pyautogui.mouseUp()
        mouse_down = False
        return InputResult(IOStatus.SUCCEEDED, "click", target=(x, y))
    except Exception as e:
        print(f"Failed to click at ({x}, {y}): {e}")
        if mouse_down:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        return InputResult(IOStatus.FAILED, "click", f"click_error:{type(e).__name__}", (x, y))


def click_relative_in_window(window_title, rel_x, rel_y, duration=0.1):
    """Simulates a mouse click relative to the specified window's top-left corner.

    Args:
        window_title: Title of the target window to locate.
        rel_x: X-coordinate relative to the window's top-left corner.
        rel_y: Y-coordinate relative to the window's top-left corner.
        duration: Mouse pointer move duration in seconds.

    Returns:
        A structured result describing success or failure.
    """
    rect = find_window_geometry(window_title)
    if rect is None:
        print(f"Error: Target window '{window_title}' could not be located. Relative click bypassed.")
        return InputResult(IOStatus.FAILED, "click", "window_not_found", (rel_x, rel_y))

    abs_x = rect["left"] + rel_x
    abs_y = rect["top"] + rel_y
    print(f"Translating relative coordinates ({rel_x}, {rel_y}) to absolute coordinates ({abs_x}, {abs_y}) for window '{window_title}'")
    return click_at(abs_x, abs_y, duration=duration)


def press_key(key, press_duration=0.05):
    """Simulates pressing and releasing a specific keyboard key.

    Args:
        key: The key to press (e.g., 'r' for reload, 'space', 'esc').
        press_duration: Duration to hold the key down in seconds.

    Returns:
        A structured result describing success or failure.
    """
    if pyautogui is None:
        print("Error: 'pyautogui' is not installed.")
        return InputResult(IOStatus.UNAVAILABLE, "key", "pyautogui_unavailable", key)

    key_down = False
    try:
        pyautogui.keyDown(key)
        key_down = True
        time.sleep(press_duration)
        pyautogui.keyUp(key)
        key_down = False
        print(f"Successfully simulated key press: '{key}'")
        return InputResult(IOStatus.SUCCEEDED, "key", target=key)
    except Exception as e:
        print(f"Failed to press key '{key}': {e}")
        if key_down:
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass
        return InputResult(IOStatus.FAILED, "key", f"key_error:{type(e).__name__}", key)


if __name__ == "__main__":
    print("Starting targeted input control test...")
    if pyautogui is not None:
        # Prevent pyautogui actions from going out of control (fail-safe)
        pyautogui.FAILSAFE = True
        
        # Test finding target window and performing relative clicks safely
        # We will attempt to simulate minor interaction on "Shotgun King" window
        target_win = "Shotgun King"
        
        print(f"Attempting relative click inside '{target_win}' window...")
        # Simulating relative click near the center of a standard 1280x720 window
        # Adjust (640, 360) safely or test a tiny movement
        click_relative_in_window(target_win, 640, 360, duration=0.5)
        
        print("Attempting to press key 'r' (reload simulation)...")
        press_key("r")
    else:
        print("Error: pyautogui is not available.")
