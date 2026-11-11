
import torch
from pathlib import Path

from src.model import GPTModel
from src.tokenizer import SimpleTokenizer


# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = Path("checkpoints/mini_gpt.pth")

# Load dataset and tokenizer
text = Path("data/cleaned_dataset.txt").read_text(
    encoding="utf-8"
)

tokenizer = SimpleTokenizer(text)

# Load trained checkpoint
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

# Initialize model
model = GPTModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    context_length=checkpoint["context_length"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# Text generation function
def generate_text(
    prompt,
    max_new_tokens=30,
    temperature=1.0,
    top_k=None
):

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("Prompt contains no known tokens.")

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the latest context tokens
            idx_cond = idx[:, -model.context_length:]

            # Forward pass
            logits = model(idx_cond)

            # Get last token predictions
            logits = logits[:, -1, :]

            # Temperature scaling
            logits = logits / temperature

            # Top-K sampling
            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[logits < values[:, [-1]]] = float("-inf")

            # Convert logits into probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append generated token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

    return tokenizer.decode(idx[0].tolist())


# Generate text
if __name__ == "__main__":

    prompt = "Artificial intelligence"

    generated_text = generate_text(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.8,
        top_k=10
    )

    print("\nInput Prompt:")
    print(prompt)

    print("\nGenerated Text:")
    print(generated_text)
