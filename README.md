# Bond_05_01 — Simplified Transformer Language Model

A from-scratch implementation of a transformer-based language model in PyTorch. The model is trained on a built-in text corpus to perform next-word prediction and autoregressive text generation.

## What It Does

- Tokenizes a text corpus and builds a vocabulary
- Trains a small transformer (multi-head self-attention + feed-forward blocks) to predict the next word in a sequence
- Generates new text autoregressively from a seed phrase
- Optionally loads a custom corpus from a file

## Architecture

| Component | Description |
|---|---|
| `PositionalEncoding` | Sinusoidal positional encodings added to token embeddings |
| `MultiHeadSelfAttention` | Scaled dot-product attention split across multiple heads |
| `TransformerBlock` | One attention sublayer + one feed-forward sublayer, each with residual connections and layer norm |
| `SimplifiedTransformerLM` | Stacks N transformer blocks; predicts the next word from the last sequence position |

## Requirements

```
torch
numpy
gensim  # optional — only needed for word2vec embedding utilities
```

Install dependencies:

```bash
pip install torch numpy gensim
```

## How to Run

```bash
python Bond_05_01.py
```

This will:
1. Train the transformer on the built-in corpus (60 epochs by default)
2. Print the final training loss and vocabulary size
3. Run several sample next-word predictions
4. Generate a 12-word continuation from the seed phrase `"in a quiet town"`
5. Save the corpus text to `embedded_corpus.txt` in the same directory

## Configuration

Hyperparameters are set inside `main()` and can be adjusted directly:

| Parameter | Default | Description |
|---|---|---|
| `num_heads` | `2` | Number of attention heads |
| `embedding_dim` | `16` | Token embedding / model dimension |
| `sequence_length` | `4` | Number of input tokens per example |
| `num_blocks` | `2` | Number of transformer blocks |
| `num_epochs` | `60` | Training epochs |
| `batch_size` | `8` | Mini-batch size |
| `learning_rate` | `1e-3` | Adam learning rate |

## Using a Custom Corpus

Pass a file path to `train_transformer`:

```python
model, vocab, idx_to_word, history = train_transformer(
    corpus_path="my_corpus.txt",
    num_heads=2,
    embedding_dim=32,
    sequence_length=5,
    num_blocks=2,
)
```

Or pass a string directly via `corpus_text`.

## Key Functions

| Function | Description |
|---|---|
| `train_transformer(...)` | Full training pipeline; returns model, vocab, idx_to_word, loss history |
| `predict_next_word(model, vocab, idx_to_word, input_words)` | Predict the single most likely next word |
| `generate_text(model, vocab, idx_to_word, seed_words, num_words_to_generate)` | Autoregressively generate text from a seed |
| `build_vocab_from_corpus(corpus)` | Tokenize text and build word↔index mappings |

## Author

Jackson Bond
