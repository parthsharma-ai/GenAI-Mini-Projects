# Mini GPT — Language Model from Scratch

A beginner-friendly implementation of a GPT-style decoder-only Transformer using Python and PyTorch.

## Project Overview

This project implements a small autoregressive language model that learns patterns from a text dataset and generates text using next-token prediction.

The objective is to understand the internal components of Large Language Models by implementing them from scratch.

## Architecture

Raw Text → Tokenizer → Token IDs → Token Embeddings + Positional Embeddings → Transformer Blocks → Language Modeling Head → Next Token Prediction → Generated Text

## Concepts Implemented

1. Text preprocessing and dataset preparation
2. Word-level tokenization
3. Vocabulary creation and token-to-ID mapping
4. Next-token prediction dataset
5. Token embeddings
6. Learned positional embeddings
7. Query, Key and Value projections
8. Scaled dot-product attention
9. Causal masking
10. Multi-head attention
11. Feed Forward Network
12. Residual connections
13. Layer normalization
14. Dropout and GELU activation
15. GPT-style decoder architecture
16. Cross-entropy loss and backpropagation
17. AdamW optimizer
18. Autoregressive text generation
19. Temperature and Top-K sampling

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

## Project Structure

- `data/` — Training dataset
- `src/` — Tokenizer, embeddings, attention and model architecture
- `checkpoints/` — Trained model checkpoint
- `prepare_data.py` — Dataset preprocessing
- `train.py` — Model training
- `generate.py` — Text generation
- `test_model.py` — Model validation

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd 01-Mini-LLM-From-Scratch
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Prepare the dataset:

```bash
python prepare_data.py
```

Train the model:

```bash
python train.py
```

Generate text:

```bash
python generate.py
```

Run tests:

```bash
python test_model.py
```

## Current Limitations

- Uses a small educational text dataset.
- Implements a simple word-level tokenizer.
- Model is intended for learning, not general-purpose language generation.
- Does not use pretrained LLM weights.
- Does not implement production-scale training optimizations.

## Learning Objective

Build a practical understanding of the internal architecture and training pipeline of a GPT-style language model through hands-on implementation.

## Future Improvements

- Implement Byte Pair Encoding (BPE).
- Train on a larger text corpus.
- Add validation and perplexity evaluation.
- Experiment with different Transformer configurations.
- Implement more advanced sampling strategies.

---

**Author:** Parth Sharma

**Project:** Generative AI — Project 01
