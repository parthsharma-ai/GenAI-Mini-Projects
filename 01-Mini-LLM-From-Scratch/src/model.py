
import torch
import torch.nn as nn

from src.transformer import TransformerBlock


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        context_length=64,
        dropout=0.1
    ):
        super().__init__()

        self.context_length = context_length

        # Token Embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional Embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        # Multiple Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                context_length,
                dropout
            )
            for _ in range(num_layers)
        ])

        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Language Modeling Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):

        B, T = idx.shape

        # Token Embeddings
        token_vectors = self.token_embedding(idx)

        # Positional Embeddings
        positions = torch.arange(
            T,
            device=idx.device
        )

        position_vectors = self.position_embedding(positions)

        # Combine embeddings
        x = token_vectors + position_vectors

        x = self.dropout(x)

        # Pass through Transformer Blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Generate vocabulary logits
        logits = self.lm_head(x)

        return logits


# Testing Complete GPT Model
if __name__ == "__main__":

    vocab_size = 100
    batch_size = 2
    context_length = 16
    embedding_dim = 64
    num_heads = 4
    num_layers = 2

    # Create sample input
    x = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length)
    )

    # Initialize GPT Model
    model = GPTModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        context_length=context_length
    )

    # Forward pass
    logits = model(x)

    print("Input Shape:", x.shape)
    print("Logits Shape:", logits.shape)

    print(
        "Total Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nComplete Mini GPT Model Working Successfully!")
