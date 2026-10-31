"""Conservative loaded-shell proposals backed by reviewed HUD references."""

import cv2

from ..vision.ammo import _count_red_slots


def loaded_shell_count(image):
    """Count the loaded row only, excluding reserve shells and grenades."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _count_red_slots(rgb[52:72, 384:544], slot_count=10)


def match_loaded(image, examples):
    """Require a reviewed matching count and recognizable surrounding HUD.

    A blank image must never become a zero-ammunition proposal. References
    must themselves agree with the shell-bar counter before they are used.
    """
    count = loaded_shell_count(image)
    if count > 2:
        return None
    # The gun rotates during aiming; compare the two shell slots independently.
    # Still require a visible gun so an empty desktop crop cannot imply zero.
    gun = image[34:75, 416:480]
    query = cv2.cvtColor(image[30:79, 378:420], cv2.COLOR_BGR2GRAY)
    if query.std() < 5 or (gun.max(axis=2) > 150).sum() < 20:
        return None
    scores = [
        float(cv2.matchTemplate(query, template[:, :34], cv2.TM_CCOEFF_NORMED).max())
        for value, template in examples
        if value == count and template.shape == (41, 168) and template[:, :34].std() >= 5
    ]
    if count == 0:
        # Empty slots are dim: the moving gun edge above/right of them can
        # dominate whole-row correlation. Verify the slot bodies separately.
        empty_query = cv2.cvtColor(image[36:76, 380:418], cv2.COLOR_BGR2GRAY)
        for value, template in examples:
            if value != 0 or template.shape != (41, 168):
                continue
            empty_template = template[6:38, 2:32]
            if empty_template.std() < 5:
                continue
            scores.append(float(cv2.matchTemplate(empty_query, empty_template, cv2.TM_CCOEFF_NORMED).max()))
    if not scores or max(scores) < .97:
        return None
    return count, max(scores)
