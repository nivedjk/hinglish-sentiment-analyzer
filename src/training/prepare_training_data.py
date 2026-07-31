import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/hinglish_clean.csv")
OUTPUT = Path("data/processed")

df = pd.read_csv(INPUT)

print("=" * 60)
print("Preparing Training Dataset")
print("=" * 60)

# Encode labels
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["label"].map(label_map)

print("\nEncoded Labels:")
print(df["label"].value_counts())

# Train (80%) + Temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Validation (10%) + Test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv(OUTPUT / "train.csv", index=False)
val_df.to_csv(OUTPUT / "val.csv", index=False)
test_df.to_csv(OUTPUT / "test.csv", index=False)

print("\nDataset Split Complete\n")

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")