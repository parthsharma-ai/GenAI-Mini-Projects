
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.tokenizer import SimpleTokenizer
from src.model import GPTModel


# Configuration
torch.manual_seed(42)

context_length = 16
batch_size = 8
embedding_dim = 64
num_heads = 4
num_layers = 2
learning_rate = 0.001
epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Training Device:", device)


# Load dataset
text = Path("data/cleaned_dataset.txt").read_text(
    encoding="utf-8"
)

tokenizer = SimpleTokenizer(text)

token_ids = tokenizer.encode(text)

vocab_size = len(tokenizer.vocab)

print("Vocabulary Size:", vocab_size)
print("Total Tokens:", len(token_ids))


# Create training dataset
class LanguageDataset(Dataset):

    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.context_length
        ]

        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


dataset = LanguageDataset(token_ids, context_length)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)


# Initialize GPT model
model = GPTModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    context_length=context_length
).to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# Training loop
model.train()

for epoch in range(epochs):

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        # Forward propagation
        logits = model(x)

        # Calculate loss
        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )


# Save trained model
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers
    },
    checkpoint_dir / "mini_gpt.pth"
)

print("\nTraining Completed Successfully!")
print("Model saved at checkpoints/mini_gpt.pth")
