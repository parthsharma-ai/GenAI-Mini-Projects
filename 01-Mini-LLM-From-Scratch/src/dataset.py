
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import SimpleTokenizer
from pathlib import Path


# Step 1: Load dataset
text = Path("data/cleaned_dataset.txt").read_text(
    encoding="utf-8"
)

# Step 2: Create tokenizer
tokenizer = SimpleTokenizer(text)

# Step 3: Convert complete text into token IDs
token_ids = tokenizer.encode(text)

print("Total tokens:", len(token_ids))
print("First 20 token IDs:", token_ids[:20])


# Step 4: Create custom dataset
class NextTokenDataset(Dataset):

    def __init__(self, token_ids, context_length):

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):

        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):

        x = self.token_ids[
            index : index + self.context_length
        ]

        y = self.token_ids[
            index + 1 : index + self.context_length + 1
        ]

        return torch.tensor(x), torch.tensor(y)


# Step 5: Configure context length
context_length = 4

dataset = NextTokenDataset(
    token_ids,
    context_length
)

# Step 6: Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Step 7: Display one batch
x, y = next(iter(loader))

print("\nInput Batch (X):")
print(x)

print("\nTarget Batch (Y):")
print(y)

print("\nInput Shape:", x.shape)
print("Target Shape:", y.shape)
