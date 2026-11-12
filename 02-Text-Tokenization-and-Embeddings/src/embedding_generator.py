
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        print("Loading sentence embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully!")

    def generate_embedding(self, text):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    def generate_multiple_embeddings(self, sentences):

        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def display_embedding(self, text):

        embedding = self.generate_embedding(text)

        print("\n" + "=" * 60)
        print("SENTENCE EMBEDDING")
        print("=" * 60)

        print("\nOriginal Sentence:")
        print(text)

        print("\nEmbedding Shape:")
        print(embedding.shape)

        print("\nFirst 10 Embedding Values:")
        print(embedding[:10])

        print("\nTotal Dimensions:")
        print(len(embedding))

        return embedding


if __name__ == "__main__":

    generator = EmbeddingGenerator()

    sentence = "Artificial intelligence is changing technology."

    generator.display_embedding(sentence)
