
import torch
import torch.nn as nn

from src.tokenizer import SimpleTokenizer
from pathlib import Path


# Step 1: Load dataset
text = Path("data/cleaned_dataset.txt").read_text(
    encoding="utf-8"
)

tokenizer = SimpleTokenizer(text)

# Step 2: Configure model dimensions
vocab_size = len(tokenizer.vocab)
context_length = 8
embedding_dim = 64

batch_size = 2

# Step 3: Create sample token IDs
token_ids = torch.tensor([
    tokenizer.encode("Artificial intelligence is transforming"),
    tokenizer.encode("Machine learning uses neural")
])

print("Token ID shape:", token_ids.shape)


# Step 4: Create token embedding layer
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Step 5: Create positional embedding layer
position_embedding = nn.Embedding(
    num_embeddings=context_length,
    embedding_dim=embedding_dim
)

# Step 6: Generate position IDs
positions = torch.arange(
    token_ids.shape[1]
)

# Step 7: Get token embeddings
token_vectors = token_embedding(token_ids)

# Step 8: Get positional embeddings
position_vectors = position_embedding(positions)

# Step 9: Combine both embeddings
final_embeddings = token_vectors + position_vectors

# Step 10: Display results
print("\nToken Embedding Shape:", token_vectors.shape)
print("Position Embedding Shape:", position_vectors.shape)
print("Final Embedding Shape:", final_embeddings.shape)

print("\nSample Token Embedding:")
print(token_vectors[0, 0])

print("\nSample Final Embedding:")
print(final_embeddings[0, 0])
