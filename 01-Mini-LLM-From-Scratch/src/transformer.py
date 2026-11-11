
import torch
import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention


# Feed Forward Network
class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.network(x)


# Transformer Block
class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):

        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            context_length,
            dropout
        )

        # Feed Forward Network
        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # First sub-block: Attention
        normalized = self.norm1(x)

        attention_output, weights = self.attention(
            normalized
        )

        x = x + self.dropout(attention_output)

        # Second sub-block: Feed Forward
        normalized = self.norm2(x)

        ff_output = self.feed_forward(normalized)

        x = x + ff_output

        return x, weights


# Testing Transformer Block
if __name__ == "__main__":

    batch_size = 2
    context_length = 8
    embedding_dim = 64
    num_heads = 4

    # Sample input
    x = torch.randn(
        batch_size,
        context_length,
        embedding_dim
    )

    # Create Transformer Block
    block = TransformerBlock(
        embedding_dim,
        num_heads,
        context_length
    )

    # Forward pass
    output, weights = block(x)

    print("Input Shape:", x.shape)
    print("Transformer Output Shape:", output.shape)
    print("Attention Weights Shape:", weights.shape)

    print("\nTransformer Block Working Successfully!")
