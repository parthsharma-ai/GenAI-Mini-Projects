
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class EmbeddingVisualizer:

    def __init__(self, embedding_generator):

        self.embedding_generator = embedding_generator

    def visualize_embeddings(self, sentences):

        if len(sentences) < 2:
            raise ValueError("At least 2 sentences are required.")

        # Generate sentence embeddings
        embeddings = (
            self.embedding_generator.generate_multiple_embeddings(
                sentences
            )
        )

        # Reduce dimensions from 384 to 2
        pca = PCA(n_components=2)

        reduced_embeddings = pca.fit_transform(embeddings)

        # Create visualization
        plt.figure(figsize=(12, 8))

        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            s=120
        )

        # Add sentence labels
        for i, sentence in enumerate(sentences):

            plt.annotate(
                f"S{i + 1}: {sentence[:35]}",
                (
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1]
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9
            )

        plt.title("Sentence Embeddings Visualization using PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save visualization
        os.makedirs("outputs", exist_ok=True)

        output_path = "outputs/embedding_visualization.png"

        plt.savefig(output_path, dpi=300)

        print("\nEmbedding visualization saved at:")
        print(output_path)

        print("\nExplained variance ratio:")
        print(pca.explained_variance_ratio_)

        plt.show()

        return reduced_embeddings
