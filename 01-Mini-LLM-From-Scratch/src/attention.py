
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, context_length):

        super().__init__()

        # Query, Key and Value layers
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.output = nn.Linear(embedding_dim, embedding_dim)

        # Causal attention mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):

        B, T, C = x.shape

        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(C)

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Softmax
        weights = torch.softmax(scores, dim=-1)

        # Weighted value aggregation
        context = weights @ V

        # Output projection
        output = self.output(context)

        return output, weights


if __name__ == "__main__":

    batch_size = 2
    context_length = 4
    embedding_dim = 64

    # Sample input
    x = torch.randn(
        batch_size,
        context_length,
        embedding_dim
    )

    # Initialize attention
    attention = SelfAttention(
        embedding_dim,
        context_length
    )

    # Forward pass
    output, weights = attention(x)

    print("Input Shape:", x.shape)
    print("Attention Output Shape:", output.shape)
    print("Attention Weights Shape:", weights.shape)

    print("\nAttention Weights:")
    print(weights[0].detach())

    print("\nSelf-Attention Working Successfully!")
