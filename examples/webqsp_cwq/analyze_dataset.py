from datasets import load_dataset

dataset = load_dataset(f"rmanluo/RoG-webqsp", split="test")

items = []
for sample in dataset:
    if len(sample["a_entity"]) > 20:
        items.append(sample)

print(f"Number of items with more than 20 answers: {len(items)}")


items = []
for sample in dataset:
    if len(sample["a_entity"]) > 30:
        items.append(sample)

print(f"Number of items with more than 30 answers: {len(items)}")


items = []
for sample in dataset:
    if len(sample["a_entity"]) > 40:
        items.append(sample)

print(f"Number of items with more than 40 answers: {len(items)}")


items = []
for sample in dataset:
    if len(sample["a_entity"]) > 50:
        items.append(sample)

print(f"Number of items with more than 50 answers: {len(items)}")


items = []
for sample in dataset:
    if len(sample["a_entity"]) > 100:
        items.append(sample)

print(f"Number of items with more than 100 answers: {len(items)}")
