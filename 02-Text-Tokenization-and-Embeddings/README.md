# Project 02: Text Tokenization & Embeddings Lab

A hands-on exploration of how machines convert human language into numerical representations and use those representations to understand meaning, compare information, and retrieve relevant content.

## Why I Built This Project

Modern Generative AI applications do much more than simply process words. Systems such as ChatGPT, semantic search engines, recommendation systems, and Retrieval-Augmented Generation (RAG) applications need to understand the meaning and context behind human language.

I built this project to understand what happens behind the scenes when text enters an AI system — how it gets broken into tokens, converted into numerical vectors, and transformed into meaningful representations that machines can compare and process.

Rather than treating embeddings as a black-box feature provided by an API, I wanted to explore the complete process practically, experiment with different tokenization techniques, generate sentence embeddings, and understand how semantic similarity enables intelligent information retrieval.

## What Is the Purpose?

The main goal is to build a practical foundation for more advanced Generative AI systems.

Through this project, I explore how an AI application can identify relationships between sentences even when they use different words but express similar meanings.

For example, a traditional keyword search may struggle to connect:

- "How can computers learn from data?"
- "Explain machine learning."

However, an embedding-based semantic search system can identify their underlying relationship by comparing their numerical representations.

This is an important capability behind modern semantic search, recommendation engines, document retrieval systems, and RAG pipelines.

## Real-World Impact and Applications

The concepts implemented in this project are useful in several real-world applications:

- **Semantic Search:** Finding relevant information based on meaning rather than exact keyword matches.
- **RAG Systems:** Retrieving relevant documents before passing contextual information to an LLM.
- **Recommendation Engines:** Finding products, articles, or content with similar semantic representations.
- **Chatbots:** Matching user questions with relevant information from a knowledge base.
- **Document Intelligence:** Comparing, organizing, and retrieving information from large collections of text.
- **AI Agents:** Helping agents search their knowledge sources and retrieve relevant context before taking actions.

Understanding these fundamentals helps me build more reliable and efficient GenAI applications instead of depending entirely on ready-made tools and frameworks.

## What I Built

This project brings together multiple components into one interactive Tokenization and Embedding Lab.

### 1. Tokenizer Comparison

Compare three approaches to tokenization:

- Custom word-level tokenizer
- BERT WordPiece tokenizer
- GPT-2 Byte-Level BPE tokenizer

Inspect how text is divided into tokens and how token IDs and attention masks are generated.

### 2. Sentence Embedding Generation

Generate dense numerical representations of sentences using the `all-MiniLM-L6-v2` Sentence Transformer model.

Each sentence is represented as a 384-dimensional vector that captures semantic information.

### 3. Semantic Similarity

Compare sentences using cosine similarity to identify how closely related their meanings are, even when their wording differs.

### 4. Embedding Visualization

Apply Principal Component Analysis (PCA) to reduce high-dimensional embeddings into two dimensions and visualize relationships between sentences.

### 5. Batch Embedding and CSV Export

Generate embeddings for multiple sentences together and export the results into CSV format for further analysis and reuse.

### 6. Interactive Semantic Search

Build a search engine that converts a query into an embedding, compares it against a collection of sentence embeddings, and retrieves the most relevant results using top-k similarity ranking.

### 7. Automated Testing

Validate tokenization, embedding generation, similarity calculations, batch processing, and semantic search through automated tests.

## Technology Stack

- Python
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Model Used

**Sentence Transformer:** `sentence-transformers/all-MiniLM-L6-v2`

- Embedding dimension: 384
- Produces dense sentence representations.
- Used for semantic similarity and semantic search.

## Project Structure

```text
02-Text-Tokenization-and-Embeddings/
├── data/
│   └── sample_sentences.txt
├── outputs/
│   ├── embedding_visualization.png
│   └── batch_embeddings.csv
├── src/
│   ├── __init__.py
│   ├── tokenizer_comparison.py
│   ├── embedding_generator.py
│   ├── similarity.py
│   ├── visualization.py
│   ├── batch_processor.py
│   └── semantic_search.py
├── main.py
├── test_project.py
├── requirements.txt
└── README.md
```

## Installation and Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows CMD:

```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Application

```bash
python main.py
```

The application provides an interactive menu to explore the different functionalities.

## Run Automated Tests

```bash
python test_project.py
```

## Generated Outputs

- `outputs/embedding_visualization.png`
- `outputs/batch_embeddings.csv`

## Key Learning Outcomes

By completing this project, I gained practical experience in:

- Understanding word-level and subword tokenization.
- Exploring how token IDs and attention masks are generated.
- Generating and working with dense sentence embeddings.
- Measuring semantic relationships using cosine similarity.
- Reducing embedding dimensions using PCA.
- Processing embeddings in batches.
- Building a basic embedding-based semantic search engine.
- Understanding the foundations required for vector databases and RAG systems.

## Limitations and Future Improvements

This implementation uses a small sample text collection and stores embeddings in memory. It is designed as a foundational learning project rather than a production-ready search platform.

Future improvements could include integrating a persistent vector database such as FAISS or ChromaDB, processing larger document collections, implementing document chunking, and building a complete RAG pipeline using an LLM.

## Part of My Generative AI Learning Journey

This is **Project 02** in my collection of 15 practical Generative AI projects.

My objective is to progressively build a strong understanding of LLMs, embeddings, vector databases, RAG architectures, and Agentic AI systems through hands-on implementations.

**Author:** Parth Sharma

**GitHub:** [parthsharma-ai](https://github.com/parthsharma-ai)

**Repository:** [GenAI-Mini-Projects](https://github.com/parthsharma-ai/GenAI-Mini-Projects)