import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "google/muril-base-cased"


class HinglishDataset(Dataset):

    def __init__(self, csv_file, max_length=128):

        self.data = pd.read_csv(csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = str(self.data.iloc[idx]["clean_text"])

        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
    