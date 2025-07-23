from pathlib import Path

PHONEME_PATH = Path(__file__).parent
DATA_DIR = PHONEME_PATH.parent / "data"
FIGURE_DIR = PHONEME_PATH.parent / "figures"
SUBMISSIONS_DIR = PHONEME_PATH.parent / "submissions"
TRAINING_ARTIFACTS_DIR = DATA_DIR / "training_artifacts"

# this should be changed based on system
NUM_WORKERS = 6
BATCH_SIZE = 64

# taken from the notebook
SENSORS_SPEECH_MASK = [
    18,
    20,
    22,
    23,
    45,
    120,
    138,
    140,
    142,
    143,
    145,
    146,
    147,
    149,
    175,
    176,
    177,
    179,
    180,
    198,
    271,
    272,
    275,
]

VISUALIZE_SENSORS_MASK = [20, 140, 176, 272]
