"""Explicit runtime modes that separate control, collection, and learning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeMode:
    """Behavioral permissions for one runtime session."""

    name: str
    interactive: bool
    learning_enabled: bool
    checkpoint_writes: bool
    requires_checkpoint: bool
    initial_epsilon: float
    decay_epsilon: bool


def resolve_runtime_mode(name, collect_epsilon=0.05):
    """Return explicit permissions for one supported runtime mode."""
    if name == "autonomous":
        raise ValueError(
            "Mode 'autonomous' is ambiguous; use collect, evaluate, or train"
        )
    if not 0.0 <= float(collect_epsilon) <= 1.0:
        raise ValueError("collect_epsilon must be between 0 and 1")
    modes = {
        "suggest": RuntimeMode("suggest", True, False, False, True, 0.0, False),
        "collect": RuntimeMode(
            "collect", False, False, False, False, float(collect_epsilon), False
        ),
        "evaluate": RuntimeMode("evaluate", False, False, False, True, 0.0, False),
        "train": RuntimeMode("train", False, True, True, False, 1.0, True),
    }
    try:
        return modes[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported mode {name!r}; use suggest, collect, evaluate, or train"
        ) from error
