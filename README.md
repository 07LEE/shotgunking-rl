# Shotgun King RL

![Shotgun King Gameplay](docs/assets/image.png)

This project is rebuilding a screen-based Shotgun King agent with reviewed gameplay data. The maintained workflow imports screenshots, supports local annotation review, and builds datasets for piece and screen-state recognition.

The live-control and DQN runtime is experimental. Reliable screen recognition and environment transitions are being established before further training.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

The project uses standard Python packaging and does not require a specific package manager.

## Data workflow

1. Put private screenshots in `data/collection/inbox/`.
2. Import and prepare one play session with `.venv/bin/python tools/preprocess/prepare_collection.py --session SESSION_ID`.
3. Review board labels and page information with `.venv/bin/python tools/review/review_collection.py --open`.
4. Build an immutable dataset with `.venv/bin/python tools/preprocess/build_piece_dataset.py --output data/collection/datasets/NAME`.

Screenshots from the same play session must use one session ID and one train, validation, or test split.

Piece templates, screenshots, annotations, datasets, and model checkpoints remain local and are excluded from Git. Only documentation assets under `docs/assets/` are published.

## Experimental runtime

The legacy runtime requires Shotgun King in a visible 1280x720 window:

```bash
.venv/bin/python train.py
```

Move the mouse pointer to the top-left corner to trigger the emergency stop. Data tools do not start reinforcement-learning training.
