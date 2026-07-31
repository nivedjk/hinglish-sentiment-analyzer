from pathlib import Path

# ==============================
# Model
# ==============================

MODEL_NAME = "google/muril-base-cased"
NUM_LABELS = 3

# ==============================
# Training
# ==============================

MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# ==============================
# Paths
# ==============================

BASE_DIR = Path(".")

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "val.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "models" / "muril_hinglish"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)