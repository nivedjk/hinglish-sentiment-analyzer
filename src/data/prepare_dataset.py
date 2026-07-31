from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_conll(filepath):
    tweets = []

    words = []
    label = None
    tweet_id = None

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line == "":

                if words:
                    tweets.append(
                        {
                            "tweet_id": tweet_id,
                            "text": " ".join(words),
                            "label": label
                        }
                    )

                words = []
                label = None
                tweet_id = None
                continue

            parts = line.split("\t")

            if parts[0] == "meta":

                tweet_id = parts[1]
                label = parts[2]

            else:

                words.append(parts[0])

    return pd.DataFrame(tweets)


print("=" * 60)
print("Loading SemEval SentiMix Dataset")
print("=" * 60)

train = parse_conll(RAW_DIR / "train_14k_split_conll.txt")
dev = parse_conll(RAW_DIR / "dev_3k_split_conll.txt")
test = parse_conll(RAW_DIR / "test_labels_hinglish.txt")

dataset = pd.concat(
    [train, dev, test],
    ignore_index=True
)

print("\nDataset Loaded Successfully")

print(f"\nTraining Samples : {len(train)}")
print(f"Validation Samples : {len(dev)}")
print(f"Test Samples : {len(test)}")
print(f"\nTotal Samples : {len(dataset)}")

print("\nSentiment Distribution\n")

print(dataset["label"].value_counts())

dataset.to_csv(
    OUTPUT_DIR / "hinglish_master.csv",
    index=False
)

print("\nDataset saved successfully!")

print(dataset.head())