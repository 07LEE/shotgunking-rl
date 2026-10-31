"""Reviewed reserve-shell proposals that exclude neighboring HUD icons."""

import cv2

from ..vision.ammo import _count_red_slots


def reserve_shell_count(image):
    """Count consecutive tall red bars, rejecting shorter grenade marks."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _count_red_slots(rgb[89:108, 384:704], slot_count=20)


def match_reserve(image, examples):
    """Verify every counted shell against reviewed individual shell shapes."""
    count = reserve_shell_count(image)
    width = max(16, count * 16)
    gun = image[34:75, 416:480]
    if (gun.max(axis=2) > 150).sum() < 20:
        return None
    # Ignore icons following the final shell; their artwork may change.
    query = cv2.cvtColor(image[74:115, 378:382 + width + 4], cv2.COLOR_BGR2GRAY)
    if query.std() < 5:
        return None
    templates = []
    for value, template in examples:
        if template.shape != (33, 324):
            continue
        indices = range(value) if count else range(1) if value == 0 else ()
        for index in indices:
            slot = template[:, index * 16:(index + 1) * 16]
            if slot.std() >= 5:
                templates.append(slot)
    if not templates:
        return None
    scores = []
    for index in range(max(1, count)):
        slot = query[:, index * 16:index * 16 + 24]
        score = max(float(cv2.matchTemplate(slot, template, cv2.TM_CCOEFF_NORMED).max()) for template in templates)
        if score < .97:
            return None
        scores.append(score)
    return count, min(scores)
