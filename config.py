import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_fixed.xlsx")
VAL_DATA_PATH = os.path.join(BASE_DIR, "validation_fixed.xlsx")
TEST_DATA_PATH = os.path.join(BASE_DIR, "unlabeled_fixed.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Model
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# Training Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

# Aspects
ASPECTS = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience",
    "general", "none"
]

# Sentiment Labels
SENTIMENT_LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_LABELS.items()}
