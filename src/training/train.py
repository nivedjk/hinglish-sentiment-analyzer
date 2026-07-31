import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ----------------------------------------------------
# Allow project imports
# ----------------------------------------------------

sys.path.append(os.path.abspath("."))

from src.config.config import *

from src.utils.logger import get_logger
from src.utils.device import get_device
from src.utils.seed import set_seed

# ----------------------------------------------------
# Initialization
# ----------------------------------------------------

set_seed(RANDOM_SEED)

logger = get_logger()

device = get_device()

logger.info("=" * 60)
logger.info("HINGLISH SENTIMENT ANALYSIS TRAINING")
logger.info("=" * 60)

start_time = time.time()

# ----------------------------------------------------
# Load CSV
# ----------------------------------------------------

logger.info("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)

val_df = pd.read_csv(VAL_PATH)

logger.info(f"Training Samples : {len(train_df)}")

logger.info(f"Validation Samples : {len(val_df)}")

# ----------------------------------------------------
# Convert to HF Dataset
# ----------------------------------------------------

train_dataset = Dataset.from_pandas(train_df)

val_dataset = Dataset.from_pandas(val_df)

# ----------------------------------------------------
# Tokenizer
# ----------------------------------------------------

logger.info("Loading MuRIL tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ----------------------------------------------------
# Tokenization
# ----------------------------------------------------

def tokenize(batch):

    return tokenizer(

        batch["clean_text"],

        truncation=True,

        max_length=MAX_LENGTH,

    )

logger.info("Tokenizing datasets...")

train_dataset = train_dataset.map(

    tokenize,

    batched=True,

)

val_dataset = val_dataset.map(

    tokenize,

    batched=True,

)

# ----------------------------------------------------
# Dynamic Padding
# ----------------------------------------------------

data_collator = DataCollatorWithPadding(

    tokenizer=tokenizer

)

# ----------------------------------------------------
# Torch Format
# ----------------------------------------------------

train_dataset.set_format(

    type="torch",

    columns=[

        "input_ids",

        "attention_mask",

        "label",

    ],

)

val_dataset.set_format(

    type="torch",

    columns=[

        "input_ids",

        "attention_mask",

        "label",

    ],

)

logger.info("Dataset ready.")

# ----------------------------------------------------
# Metrics
# ----------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(

        labels,

        predictions,

    )

    precision, recall, f1, _ = precision_recall_fscore_support(

        labels,

        predictions,

        average="weighted",

        zero_division=0,

    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

    }

# ----------------------------------------------------
# Model
# ----------------------------------------------------

logger.info("Loading MuRIL model...")

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_LABELS,

)

model.to(device)
# ----------------------------------------------------
# Training Arguments
# ----------------------------------------------------

training_args = TrainingArguments(

    output_dir=str(MODEL_DIR),

    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    weight_decay=WEIGHT_DECAY,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    dataloader_num_workers=0,

    report_to="none",

    seed=RANDOM_SEED,

)

logger.info("Training configuration ready.")

# ----------------------------------------------------
# Early Stopping
# ----------------------------------------------------

callbacks = [

    EarlyStoppingCallback(

        early_stopping_patience=2,

        early_stopping_threshold=0.001

    )

]

# ----------------------------------------------------
# Trainer
# ----------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=callbacks,

)

logger.info("=" * 60)
logger.info("Starting Training")
logger.info("=" * 60)

trainer.train()

logger.info("=" * 60)
logger.info("Training Finished")
logger.info("=" * 60)

# ----------------------------------------------------
# Validation Evaluation
# ----------------------------------------------------

logger.info("Evaluating best model...")

results = trainer.evaluate()

logger.info("Validation Results")

for key, value in results.items():

    logger.info(f"{key} : {value}")
# ----------------------------------------------------
# Save Best Model
# ----------------------------------------------------

logger.info("=" * 60)
logger.info("Saving model...")
logger.info("=" * 60)

trainer.save_model(str(MODEL_DIR))

tokenizer.save_pretrained(str(MODEL_DIR))

logger.info(f"Model saved to: {MODEL_DIR}")

# ----------------------------------------------------
# Save Training Metrics
# ----------------------------------------------------

metrics_file = OUTPUT_DIR / "training_metrics.txt"

with open(metrics_file, "w", encoding="utf-8") as f:

    f.write("========== TRAINING RESULTS ==========\n\n")

    for key, value in results.items():
        f.write(f"{key}: {value}\n")

logger.info(f"Metrics saved to: {metrics_file}")

# ----------------------------------------------------
# Training Summary
# ----------------------------------------------------

end_time = time.time()

elapsed = end_time - start_time

hours = int(elapsed // 3600)

minutes = int((elapsed % 3600) // 60)

seconds = int(elapsed % 60)

logger.info("=" * 60)
logger.info("TRAINING SUMMARY")
logger.info("=" * 60)

logger.info(f"Model           : {MODEL_NAME}")
logger.info(f"Epochs          : {EPOCHS}")
logger.info(f"Batch Size      : {BATCH_SIZE}")
logger.info(f"Learning Rate   : {LEARNING_RATE}")

logger.info(f"Training Samples: {len(train_df)}")
logger.info(f"Validation      : {len(val_df)}")

logger.info(
    f"Training Time   : {hours}h {minutes}m {seconds}s"
)

logger.info("=" * 60)
logger.info("Training completed successfully!")
logger.info("=" * 60)

print("\n")
print("=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)
print(f"Saved Model : {MODEL_DIR}")
print(f"Metrics     : {metrics_file}")
print("=" * 60)