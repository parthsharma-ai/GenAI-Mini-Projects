
import re
from pathlib import Path


class SimpleTokenizer:

    def __init__(self, text):

        # Split text into words and punctuation
        self.tokens = re.findall(r"\w+|[^\w\s]", text)

        # Create unique vocabulary
        self.vocab = sorted(set(self.tokens))

        # Map tokens to IDs
        self.token_to_id = {
            token: idx
            for idx, token in enumerate(self.vocab)
        }

        # Reverse mapping
        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, text):

        tokens = re.findall(r"\w+|[^\w\s]", text)

        return [
            self.token_to_id[token]
            for token in tokens
            if token in self.token_to_id
        ]

    def decode(self, token_ids):

        return " ".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )


# Load cleaned dataset
data_path = Path("data/cleaned_dataset.txt")

text = data_path.read_text(encoding="utf-8")

# Create tokenizer
tokenizer = SimpleTokenizer(text)

# Test sentence
sentence = "I love artificial intelligence."

# Encoding
encoded = tokenizer.encode(sentence)

# Decoding
decoded = tokenizer.decode(encoded)

print("Vocabulary:", tokenizer.vocab)
print("\nVocabulary Size:", len(tokenizer.vocab))
print("\nOriginal Sentence:", sentence)
print("\nEncoded IDs:", encoded)
print("\nDecoded Sentence:", decoded)

