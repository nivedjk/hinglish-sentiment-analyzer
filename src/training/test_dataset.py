from dataset import HinglishDataset

dataset = HinglishDataset("data/processed/train.csv")

print("=" * 50)

print("Dataset Size")

print(len(dataset))

print("=" * 50)

sample = dataset[0]

print(sample.keys())

print(sample["input_ids"].shape)

print(sample["attention_mask"].shape)

print(sample["labels"])