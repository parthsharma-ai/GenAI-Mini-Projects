
import os
import json
import pandas as pd


class BatchEmbeddingProcessor:

    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator

    def process_sentences(self, sentences):

        if not sentences:
            raise ValueError("Sentence list cannot be empty.")

        # Generate embeddings in one batch
        embeddings = self.embedding_generator.generate_multiple_embeddings(
            sentences
        )

        print("\nBatch embedding generation completed!")
        print("Number of sentences:", len(sentences))
        print("Embedding matrix shape:", embeddings.shape)

        return embeddings

    def export_to_csv(self, sentences, embeddings):

        if len(sentences) != len(embeddings):
            raise ValueError(
                "Number of sentences and embeddings must match."
            )

        os.makedirs("outputs", exist_ok=True)

        records = []

        for sentence, embedding in zip(sentences, embeddings):

            records.append({
                "sentence": sentence,
                "embedding_dimension": len(embedding),
                "embedding_vector": json.dumps(
                    embedding.tolist()
                )
            })

        dataframe = pd.DataFrame(records)

        output_path = "outputs/batch_embeddings.csv"

        dataframe.to_csv(output_path, index=False)

        print("\nCSV exported successfully!")
        print("File location:", output_path)
        print("Rows:", len(dataframe))
        print("Columns:", len(dataframe.columns))

        return dataframe


if __name__ == "__main__":

    from src.embedding_generator import EmbeddingGenerator

    generator = EmbeddingGenerator()

    processor = BatchEmbeddingProcessor(generator)

    sample_sentences = [
        "Artificial intelligence is transforming technology.",
        "Machine learning helps computers learn from data.",
        "I enjoy building Generative AI applications.",
        "Python is a popular programming language.",
        "The weather is pleasant today."
    ]

    embeddings = processor.process_sentences(sample_sentences)

    processor.export_to_csv(
        sample_sentences,
        embeddings
    )
