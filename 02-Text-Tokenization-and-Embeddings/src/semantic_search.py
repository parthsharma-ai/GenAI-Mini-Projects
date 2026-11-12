
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:

    def __init__(self, embedding_generator, sentences):

        if not sentences:
            raise ValueError("Search corpus cannot be empty.")

        self.embedding_generator = embedding_generator
        self.sentences = sentences

        print("\nGenerating corpus embeddings...")

        # Generate embeddings once and reuse them
        self.corpus_embeddings = (
            self.embedding_generator.generate_multiple_embeddings(
                self.sentences
            )
        )

        print("Corpus embeddings generated!")
        print("Corpus size:", len(self.sentences))
        print("Embedding shape:", self.corpus_embeddings.shape)

    def search(self, query, top_k=3):

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Generate query embedding
        query_embedding = (
            self.embedding_generator.generate_embedding(query)
        )

        # Compare query with all corpus embeddings
        scores = cosine_similarity(
            [query_embedding],
            self.corpus_embeddings
        )[0]

        # Get indices of highest similarity scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):

            result = {
                "rank": rank,
                "sentence": self.sentences[index],
                "score": float(scores[index])
            }

            results.append(result)

        return results

    def display_results(self, query, top_k=3):

        results = self.search(query, top_k)

        print("\n" + "=" * 65)
        print("SEMANTIC SEARCH RESULTS")
        print("=" * 65)

        print("\nSearch Query:")
        print(query)

        print("\nMost Relevant Results:\n")

        for result in results:

            print(f"Rank: {result['rank']}")
            print(f"Sentence: {result['sentence']}")
            print(f"Similarity: {result['score']:.4f}")
            print("-" * 50)

        return results


if __name__ == "__main__":

    from src.embedding_generator import EmbeddingGenerator

    with open("data/sample_sentences.txt", "r", encoding="utf-8") as file:
        sentences = [
            line.strip()
            for line in file
            if line.strip()
        ]

    generator = EmbeddingGenerator()

    search_engine = SemanticSearch(
        generator,
        sentences
    )

    query = "How do computers learn from information?"

    search_engine.display_results(query, top_k=3)
