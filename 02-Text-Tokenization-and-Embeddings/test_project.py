
import numpy as np

from src.tokenizer_comparison import TokenizerComparison
from src.embedding_generator import EmbeddingGenerator
from src.similarity import SemanticSimilarity
from src.batch_processor import BatchEmbeddingProcessor
from src.semantic_search import SemanticSearch


def run_tests():

    print("\n========== PROJECT 2 TEST SUITE ==========\n")

    # Initialize models once
    tokenizer = TokenizerComparison()
    generator = EmbeddingGenerator()

    # TEST 1: Tokenization
    print("\nTEST 1: TOKENIZATION")

    text = "Artificial intelligence is transforming technology."

    word_tokens = tokenizer.word_tokenize(text)
    bert_result = tokenizer.bert_tokenize(text)
    gpt2_result = tokenizer.gpt2_tokenize(text)

    assert len(word_tokens) > 0
    assert len(bert_result["token_ids"]) > 0
    assert len(gpt2_result["token_ids"]) > 0

    print("PASS: Tokenization works.")

    # TEST 2: Embedding generation
    print("\nTEST 2: EMBEDDING GENERATION")

    embedding = generator.generate_embedding(text)

    assert embedding.shape == (384,)
    assert np.isfinite(embedding).all()

    print("PASS: Embedding generated.")
    print("Embedding dimensions:", len(embedding))

    # TEST 3: Similarity
    print("\nTEST 3: SEMANTIC SIMILARITY")

    similarity = SemanticSimilarity(generator)

    score = similarity.calculate_similarity(
        "I love machine learning.",
        "I enjoy studying artificial intelligence."
    )

    assert -1.0 <= score <= 1.0

    print("PASS: Similarity calculated.")
    print("Similarity score:", round(score, 4))

    # TEST 4: Batch processing
    print("\nTEST 4: BATCH EMBEDDINGS")

    sentences = [
        "Machine learning is useful.",
        "Python is a programming language.",
        "Artificial intelligence is growing."
    ]

    processor = BatchEmbeddingProcessor(generator)

    embeddings = processor.process_sentences(sentences)

    assert embeddings.shape == (3, 384)
    assert np.isfinite(embeddings).all()

    print("PASS: Batch embeddings generated.")

    # TEST 5: Semantic search
    print("\nTEST 5: SEMANTIC SEARCH")

    search_engine = SemanticSearch(
        generator,
        sentences
    )

    results = search_engine.search(
        "What is machine learning?",
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["rank"] == 1

    print("PASS: Semantic search works.")

    # Final result
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
