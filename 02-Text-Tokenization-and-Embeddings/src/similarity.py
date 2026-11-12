
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarity:

    def __init__(self, embedding_generator):

        self.embedding_generator = embedding_generator

    def calculate_similarity(self, sentence1, sentence2):

        # Generate embeddings
        embedding1 = self.embedding_generator.generate_embedding(
            sentence1
        )

        embedding2 = self.embedding_generator.generate_embedding(
            sentence2
        )

        # Calculate cosine similarity
        score = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return float(score)

    def compare_sentences(self, sentence1, sentence2):

        score = self.calculate_similarity(
            sentence1,
            sentence2
        )

        print("\n" + "=" * 60)
        print("SEMANTIC SIMILARITY RESULT")
        print("=" * 60)

        print("\nSentence 1:")
        print(sentence1)

        print("\nSentence 2:")
        print(sentence2)

        print(f"\nSimilarity Score: {score:.4f}")

        if score >= 0.75:
            print("Interpretation: High semantic similarity")

        elif score >= 0.45:
            print("Interpretation: Moderate semantic similarity")

        else:
            print("Interpretation: Low semantic similarity")

        return score

    def compare_multiple_sentences(self, sentences):

        # Generate all embeddings together
        embeddings = (
            self.embedding_generator.generate_multiple_embeddings(
                sentences
            )
        )

        # Calculate pairwise similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        print("\n" + "=" * 60)
        print("PAIRWISE SIMILARITY MATRIX")
        print("=" * 60)

        for i, sentence in enumerate(sentences):
            print(f"\n{i + 1}. {sentence}")

        print("\nSimilarity Matrix:\n")
        print(np.round(similarity_matrix, 3))

        return similarity_matrix
