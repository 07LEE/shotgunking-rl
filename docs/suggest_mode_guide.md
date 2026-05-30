# Suggest Mode Action Shortcuts Guide

This document describes the keyboard shortcuts and action mapping layout utilized when interacting with the Console input in Suggest Mode.

## Important Technical Limitations

Because Console text inputs (via Python's input function) process standard number keystrokes and Numpad keystrokes as identical characters, the system cannot distinguish between the Numpad layout and the standard numeric row.

To prioritize directional grid movement, the numpad_map is enforced by default. Therefore, standard numeric actions (such as typing 8 for manual Reload) are overridden by the Numpad layout mapping (which redirects 8 to Move Up).

## Keyboard Input Action Mapping

The table below outlines the exact action triggered by each specific character entered in the Console prompt.

| Entered Character | Internal Action ID | Resolved Action | Description |
| :--- | :--- | :--- | :--- |
| 7 | 0 | Move Up-Left | Moves Player King 1 tile up-left |
| 8 | 1 | Move Up | Moves Player King 1 tile up |
| 9 | 2 | Move Up-Right | Moves Player King 1 tile up-right |
| 4 | 3 | Move Left | Moves Player King 1 tile left |
| 6 | 4 | Move Right | Moves Player King 1 tile right |
| 1 | 5 | Move Down-Left | Moves Player King 1 tile down-left |
| 2 | 6 | Move Down | Moves Player King 1 tile down |
| 3 | 7 | Move Down-Right | Moves Player King 1 tile down-right |
| 5 | 8 | Reload | Initiates manual weapon reload sequence |
| 0 | 9 | Shoot | Executes shooting action (targets closest threat) |
| . | - | Force Defeat | Terminates current episode under defeat condition |
| Enter (empty input) | - | Accept Recommendation | Confirms and triggers the default AI suggested action |
