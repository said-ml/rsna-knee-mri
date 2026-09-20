from pathlib import Path


PROJECT_ROOT = Path("/workspace")

DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_ROOT / "raw" / "train.csv"

ZARR_MANIFEST = (
    DATA_ROOT
    / "reports"
    / "baseline_001_series_selection.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_001"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "baseline_001"
LOG_DIR = PROJECT_ROOT / "logs" / "baseline_001"


SEED = 42

VAL_FRACTION = 0.20

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


VOLUME_SIZE = (32, 128, 128)

BATCH_SIZE = 2

NUM_WORKERS = 2

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.20

AMP = True


EXPERIMENT_NAME = "BASELINE-001"
