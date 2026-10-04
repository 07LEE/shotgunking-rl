"""Compatibility exports for the packaged vision analyzer."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("shotgun_king_rl.vision.analyzer", run_name="__main__")
else:
    from shotgun_king_rl.vision.analyzer import *  # noqa: F403
