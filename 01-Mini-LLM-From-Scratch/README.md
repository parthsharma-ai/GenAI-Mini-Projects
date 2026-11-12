# Project 1: Mini LLM From Scratch

### Building a Small GPT-Style Language Model Using PyTorch

## Overview

Large Language Models (LLMs) such as ChatGPT, Gemini, and Claude can understand and generate human-like text. But how do these models actually work internally?

I built this project to explore the fundamental architecture behind modern LLMs by implementing a small GPT-style language model from scratch using Python and PyTorch.

Instead of relying on a pretrained language model, this project implements the core components of a decoder-only Transformer architecture, including tokenization, embeddings, causal self-attention, multi-head attention, Transformer blocks, model training, and text generation.

The objective was to understand how raw text is transformed into numerical representations, how a language model learns patterns, and how it generates new text one token at a time.

## Why I Built This Project

Modern Generative AI applications are built on top of powerful language models. However, simply using an existing API does not explain how these models process information or generate responses.

I wanted to understand the internal working of an LLM at an architectural level rather than treating it as a black box.

Through this project, I explored the fundamental building blocks of GPT-style models and implemented a simplified version that can learn from a text dataset and generate new text.

## What Is the Purpose?

The primary purpose of this project is to demonstrate the working principles of a small GPT-style language model.

The model learns statistical patterns from a training corpus and uses the learned patterns to predict the next token in a sequence.

It demonstrates the basic next-token prediction process that forms the foundation of autoregressive language generation.

## What I Built

### 1. Dataset Preparation
- Created and prepared a text dataset for training.
- Cleaned and processed the raw text.
- Converted the text into a format suitable for model training.

### 2. Custom Tokenizer
- Implemented a word-level tokenizer.
- Created a vocabulary from the training corpus.
- Converted words into numerical token IDs.
- Implemented decoding to convert token IDs back into text.

### 3. Dataset and Input Preparation
- Created a custom dataset loader using PyTorch.
- Generated input and target sequences.
- Implemented shifted sequences for next-token prediction.

### 4. Token Embeddings
- Converted token IDs into dense vector representations.
- Added positional embeddings to provide information about token positions.
- Combined token and positional embeddings before passing them to the Transformer.

### 5. Causal Self-Attention
- Implemented scaled dot-product attention.
- Applied causal masking to prevent the model from accessing future tokens.
- Enabled the model to learn relationships between tokens in a sequence.

### 6. Multi-Head Attention
- Implemented multiple attention heads.
- Allowed the model to learn different relationships between tokens simultaneously.
- Combined the outputs of multiple attention heads.

### 7. Transformer Block
- Combined multi-head self-attention and feed-forward neural networks.
- Implemented residual connections and layer normalization.
- Used dropout for regularization.

### 8. GPT-Style Language Model
- Built a decoder-only Transformer architecture.
- Combined embeddings, Transformer blocks, and a language-model output layer.
- Generated next-token prediction scores.

### 9. Model Training
- Trained the model using next-token prediction.
- Used cross-entropy loss to measure prediction error.
- Used the AdamW optimizer to update model parameters.
- Saved the trained model checkpoint.

### 10. Text Generation
- Loaded the trained model checkpoint.
- Generated text autoregressively, one token at a time.
- Experimented with temperature and top-k sampling to control generation.

## Model Architecture

The project follows a simplified GPT-style decoder-only Transformer architecture.

```text
Input Text
    |
    v
Custom Tokenizer
    |
    v
Token IDs
    |
    v
Token Embeddings
    +
Positional Embeddings
    |
    v
Multi-Head Causal Self-Attention
    |
    v
Feed-Forward Neural Network
    |
    v
Transformer Blocks
    |
    v
Linear Output Layer
    |
    v
Next-Token Prediction
    |
    v
Generated Text
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Model development and training |
| NumPy | Numerical operations |
| Matplotlib | Training visualization, if used |
| Git & GitHub | Version control and project hosting |

## Project Structure

```text
01-Mini-LLM-From-Scratch/
│
├── data/
│   ├── dataset.txt
│   └── cleaned_dataset.txt
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── multi_head_attention.py
│   ├── transformer.py
│   └── model.py
│
├── checkpoints/
│   └── mini_gpt.pth
│
├── prepare_data.py
├── train.py
├── generate.py
├── test_model.py
├── requirements.txt
└── README
