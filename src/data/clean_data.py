import re
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("data/processed/hinglish_master.csv")
OUTPUT_FILE = Path("data/processed/hinglish_clean.csv")

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Cleaning Hinglish Dataset")
print("=" * 60)

print(f"Original Samples : {len(df)}")


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Remove duplicate tweets
df = df.drop_duplicates(subset=["clean_text"])

# Remove empty tweets
df = df[df["clean_text"].str.strip() != ""]

print(f"Remaining Samples : {len(df)}")

print("\nLabel Distribution:\n")
print(df["label"].value_counts())

# Save cleaned dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Clean dataset saved successfully!")
print(f"Location: {OUTPUT_FILE}")