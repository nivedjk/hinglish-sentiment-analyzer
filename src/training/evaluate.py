import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from datasets import Dataset

sys.path.append(os.path.abspath("."))

from src.config.config import *

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Evaluation Started")
print("Device:", device)
print("=" * 60)

# ============================================================
# Load Model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(VAL_PATH)

dataset = Dataset.from_pandas(df)

label_names = ["negative", "neutral", "positive"]

# ============================================================
# Tokenization
# ============================================================

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

# ============================================================
# Inference
# ============================================================

predictions = []
true_labels = []
confidences = []

print("\nRunning inference...\n")

with torch.no_grad():

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(outputs.logits, dim=1)

        conf, preds = torch.max(probs, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        confidences.extend(conf.cpu().numpy())

# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(true_labels, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="weighted",
)

print("\nAccuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=label_names,
    )
)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(true_labels, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names,
)

disp.plot(cmap="Blues")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")

print("\nSaved confusion_matrix.png")

# ============================================================
# Save Predictions
# ============================================================

id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

results = df.copy()

results["actual"] = [id_to_label[x] for x in true_labels]
results["prediction"] = [id_to_label[x] for x in predictions]

results["confidence"] = [
    round(float(c) * 100, 2)
    for c in confidences
]

results.to_csv(
    "outputs/predictions.csv",
    index=False,
)

print("Saved predictions.csv")

print("\nEvaluation Complete.")