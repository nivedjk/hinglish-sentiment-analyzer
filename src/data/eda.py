import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/processed/hinglish_clean.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Exploratory Data Analysis")
print("=" * 60)

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nLabel Distribution")
print(df["label"].value_counts())

# -----------------------
# Sentence Length
# -----------------------

df["text_length"] = df["clean_text"].apply(len)

print("\nAverage Sentence Length")

print(df["text_length"].mean())

# -----------------------
# Label Distribution Plot
# -----------------------

plt.figure(figsize=(6,5))

df["label"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig("outputs/label_distribution.png")

print("\nSaved: outputs/label_distribution.png")

# -----------------------
# Text Length Histogram
# -----------------------

plt.figure(figsize=(8,5))

plt.hist(df["text_length"], bins=40)

plt.title("Tweet Length Distribution")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/text_length_distribution.png")

print("Saved: outputs/text_length_distribution.png")