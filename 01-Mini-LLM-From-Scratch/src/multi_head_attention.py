
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Combined Query, Key and Value projection
        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        # Output projection
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(context_length, context_length)
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B, T, 3, self.num_heads, self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv.unbind(0)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Apply causal masking
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Softmax
        weights = torch.softmax(scores, dim=-1)

        weights = self.dropout(weights)

        # Weighted value aggregation
        context = weights @ V

        # Concatenate attention heads
        context = context.transpose(1, 2).contiguous()

        context = context.reshape(B, T, C)

        # Final projection
        output = self.output(context)

        return output, weights


# Testing
if __name__ == "__main__":

    batch_size = 2
    context_length = 8
    embedding_dim = 64
    num_heads = 4

    x = torch.randn(
        batch_size,
        context_length,
        embedding_dim
    )

    model = MultiHeadAttention(
        embedding_dim,
        num_heads,
        context_length
    )

    output, weights = model(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print("Attention Weights Shape:", weights.shape)

    print("\nMulti-Head Attention Working Successfully!")
