# Suggest Mode Action Shortcuts Guide

This document describes the keyboard shortcuts and action mapping layout utilized when interacting with the Console input in Suggest Mode.

Start a bounded validation session with `./scripts/run_suggest.sh`. The launcher uses one episode and at most ten actions by default, while additional command-line arguments may override these values.

## Important Technical Limitations

Because Console text inputs (via Python's input function) process standard number keystrokes and Numpad keystrokes as identical characters, the system cannot distinguish between the Numpad layout and the standard numeric row.

To prioritize directional grid movement, the numpad_map is enforced by default. Therefore, standard numeric actions (such as typing 8 for manual Reload) are overridden by the Numpad layout mapping (which redirects 8 to Move Up).

Before requesting input, Suggest Mode prints the recognized 8x8 board, loaded and reserve ammunition, every action permitted by the current action mask, and the targeting or movement rule that makes each action available. It ranks up to three valid actions by raw model Q-value. Q-values are model scores, not probabilities or calibrated confidence, and the displayed reasons explain the deterministic validity and targeting rules rather than the model's internal reasoning.

Suggest Mode requires an existing readable model checkpoint. It records the result but never updates replay memory, optimizer state, target weights or model checkpoints. Use `train` explicitly when model updates are intended.

## Runtime traces

Every attempted action in Suggest, Collect, Evaluate and Train modes is appended to a private JSONL file under `data/runtime_traces`. Each line contains the configured weapon, rank and floor, the exact board and ammunition state used for the decision, the action mask and Q-values, recommendations, the selected and executed actions, input and completion results, the next board, reward and terminal status. A matching manifest records the mode, learning and checkpoint permissions, epsilon, scope and pre-run model hash. Failed input, rejected observations and all non-training sessions remain `accepted_for_replay: false`.

Immediately before input, the runtime captures a fresh observation and compares its board and ammunition with the state used to choose the action. A mismatch blocks input, keeps the action counter unchanged and records `pre_action_observation_changed` with changed squares and ammunition values in the trace.

The trace is operational evidence, not verified perception training data. Screenshots and labels for piece, ammunition and screen readers still require the collection and human-review pipeline.

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
