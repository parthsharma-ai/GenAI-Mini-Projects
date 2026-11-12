
from src.tokenizer_comparison import TokenizerComparison
from src.embedding_generator import EmbeddingGenerator
from src.similarity import SemanticSimilarity
from src.visualization import EmbeddingVisualizer
from src.batch_processor import BatchEmbeddingProcessor
from src.semantic_search import SemanticSearch


def load_corpus():

    with open(
        "data/sample_sentences.txt",
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def main():

    print("\n==========================================")
    print("     GENAI TOKENIZATION & EMBEDDING LAB")
    print("==========================================")

    print("\nLoading models...")

    tokenizer = TokenizerComparison()
    generator = EmbeddingGenerator()

    corpus = load_corpus()

    similarity = SemanticSimilarity(generator)
    visualizer = EmbeddingVisualizer(generator)
    processor = BatchEmbeddingProcessor(generator)
    search_engine = SemanticSearch(generator, corpus)

    while True:

        print("\n========== MAIN MENU ==========")
        print("1. Tokenizer Comparison")
        print("2. Generate Sentence Embedding")
        print("3. Compare Two Sentences")
        print("4. Visualize Embeddings")
        print("5. Batch Embedding and CSV Export")
        print("6. Interactive Semantic Search")
        print("0. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":

            text = input("\nEnter text: ").strip()

            if text:
                tokenizer.compare(text)
            else:
                print("Text cannot be empty.")

        elif choice == "2":

            text = input("\nEnter sentence: ").strip()

            if text:
                generator.display_embedding(text)
            else:
                print("Text cannot be empty.")

        elif choice == "3":

            sentence1 = input("\nEnter first sentence: ").strip()
            sentence2 = input("Enter second sentence: ").strip()

            if sentence1 and sentence2:
                similarity.compare_sentences(
                    sentence1,
                    sentence2
                )
            else:
                print("Both sentences are required.")

        elif choice == "4":

            visualizer.visualize_embeddings(corpus)

        elif choice == "5":

            embeddings = processor.process_sentences(corpus)

            processor.export_to_csv(
                corpus,
                embeddings
            )

        elif choice == "6":

            while True:

                query = input(
                    "\nEnter search query (or type 'back'): "
                ).strip()

                if query.lower() == "back":
                    break

                if query:
                    search_engine.display_results(
                        query,
                        top_k=3
                    )
                else:
                    print("Query cannot be empty.")

        elif choice == "0":

            print("\nThank you for using the application!")
            break

        else:

            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
