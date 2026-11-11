
import torch
from pathlib import Path

from src.tokenizer import SimpleTokenizer
from src.model import GPTModel


# Load dataset
text = Path("data/cleaned_dataset.txt").read_text(
    encoding="utf-8"
)

tokenizer = SimpleTokenizer(text)

# Load checkpoint
checkpoint = torch.load(
    "checkpoints/mini_gpt.pth",
    map_location="cpu",
    weights_only=False
)

# Initialize model
model = GPTModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    context_length=checkpoint["context_length"]
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# Test 1: Forward pass
x = torch.randint(
    0,
    checkpoint["vocab_size"],
    (2, 8)
)

with torch.no_grad():
    output = model(x)

assert output.shape == (
    2, 8, checkpoint["vocab_size"]
)

print("Test 1: Forward Pass PASSED")

# Test 2: Output contains valid numbers
assert torch.isfinite(output).all()

print("Test 2: Numerical Stability PASSED")

# Test 3: Parameter count
parameters = sum(
    p.numel() for p in model.parameters()
)

print("Test 3: Model Parameters:", parameters)

# Test 4: Tokenizer
sample = "Artificial intelligence"
encoded = tokenizer.encode(sample)

assert len(encoded) > 0

print("Test 4: Tokenizer PASSED")

print("\nALL TESTS PASSED!")
