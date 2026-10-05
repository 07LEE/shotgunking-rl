"""HUD ammunition recognition independent from board analysis."""

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


LEGACY_AMMO_FALLBACK = (2, 8)


def read_ammo_count(img):
    """Return loaded and reserve ammunition, or ``None`` on recognition failure."""
    if img is None or cv2 is None or np is None:
        return None

    try:
        height, width = img.shape[:2]
        if height != 720 or width != 1280:
            img = cv2.resize(img, (1280, 720))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        loaded = _count_red_slots(rgb[52:72, 384:544], slot_count=10)
        reserve = _count_red_slots(rgb[89:108, 384:704], slot_count=20)
        return loaded, reserve
    except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
        return None


def extract_ammo_count(img):
    """Preserve the legacy tuple fallback for existing runtime callers."""
    reading = read_ammo_count(img)
    return reading if reading is not None else LEGACY_AMMO_FALLBACK


def _count_red_slots(region, slot_count):
    red_mask = (
        (region[:, :, 0] > 180)
        & (region[:, :, 1] < 60)
        & (region[:, :, 2] < 80)
    )
    count = 0
    for index in range(slot_count):
        start_x = index * 16
        end_x = start_x + 8
        if np.sum(red_mask[:, start_x:end_x]) < 15:
            break
        count += 1
    return count
