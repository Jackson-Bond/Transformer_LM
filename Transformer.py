from __future__ import annotations

import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_CORPUS = """In a quiet town near a river, a group of students met every evening in a small library to study science, mathematics, and literature. The library was not large, but it was filled with old wooden shelves, long tables, and soft lamps that made the room feel warm and peaceful. The students came from different backgrounds, and each person had a different reason for being there. Some wanted to become engineers, some hoped to become teachers, and others simply loved to learn.

One student, named Sara, enjoyed reading about astronomy. She often opened books about stars, planets, and the motion of galaxies. She was fascinated by the idea that light from distant stars had traveled for years before reaching Earth. Another student, Daniel, preferred mathematics. He liked solving equations and proving small theorems for fun. He believed that numbers described hidden patterns in the world. A third student, Leila, cared most about language. She collected unusual words and wrote short stories in a notebook with neat handwriting.

Every evening, the librarian placed a large clock on the front desk and reminded the students that time moved quickly. "Use your hours wisely," she would say. At first, the students laughed at the repeated advice, but over time they understood its meaning. They noticed that even a single hour of focused study could change what they understood by the end of the night.

During one winter, the town experienced many cold and rainy days. The river rose, the streets became muddy, and the sky remained gray for weeks. Because of the weather, fewer people visited the library. The students, however, continued to come. They wore heavy coats, carried wet umbrellas, and sat near the lamps to stay warm. Their conversations became longer and deeper. They spoke about history, art, machines, and the future. They debated whether intelligence was mainly the result of memory, reasoning, or imagination.

One evening, Daniel brought a small mechanical clock that no longer worked. He placed it on the table and asked whether anyone could fix it. Sara looked closely at the gears. Leila listened to the faint sound it made when shaken. Together they opened the back panel and studied the tiny springs and wheels inside. After several attempts, they managed to repair one broken piece. When the clock finally started ticking again, the students smiled as if they had solved an important scientific mystery.

That small success changed the mood of the group. They began bringing other objects to examine: an old radio, a broken flashlight, a hand mirror with a loose frame, and even a toy boat with a damaged sail. Each object became a lesson. The radio led to a discussion about signals and sound. The flashlight led to questions about batteries and circuits. The mirror inspired a conversation about light, reflection, and geometry. The toy boat brought up ideas about force, balance, and motion on water.

As spring arrived, the weather improved and the town became lively again. Flowers appeared near the sidewalks, birds returned to the trees, and the river moved more gently under the sun. The students still met at the library, but now they sometimes took their notebooks outside. They sat under a large tree and reviewed what they had learned over the winter. They discovered that knowledge grew best when curiosity, patience, and effort worked together.

By the end of the season, each student had changed in some way. Sara had become more confident in explaining scientific ideas. Daniel had learned that mathematics was not only about correct answers but also about clear thinking. Leila had found that language could connect art and science in surprising ways. They all realized that learning was not a race to finish first. It was a gradual process of observation, practice, error, and improvement.

Years later, each of them would remember the library, the lamps, the rain, and the long evenings of conversation. They would remember the repaired clock and the quiet feeling of discovery that followed each new idea. Most of all, they would remember that understanding often begins with a simple question, asked carefully, and pursued with patience."""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_vocab_from_corpus(corpus, min_freq=1):
    """
    Tokenize raw text and build vocabulary mappings.

    Parameters
    ----------
    corpus : str
        Input text corpus.
    min_freq : int
        Minimum frequency threshold.

    Returns
    -------
    tokens : list[str]
    vocab : dict[str, int]
    idx_to_word : dict[int, str]
    """
    corpus = corpus.lower()
    corpus = re.sub(r"[^a-z0-9\s]", " ", corpus)
    tokens = corpus.split()

    counts = Counter(tokens)
    vocab_words = sorted([w for w, c in counts.items() if c >= min_freq])
    vocab = {word: idx for idx, word in enumerate(vocab_words)}
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return tokens, vocab, idx_to_word


def train_word2vec_embeddings(tokens, embedding_dim=16, window=5, min_count=1):
    """
    Train a gensim Word2Vec model on the provided tokens.
    """
    try:
        from gensim.models import Word2Vec
    except Exception as e:
        raise ImportError(
            "gensim is required for train_word2vec_embeddings. Install it with pip install gensim"
        ) from e

    sentences = [tokens]
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=1,
        seed=42,
    )
    return model


def get_word_embedding(word, embedding_dim, gensim_model=None):
    """
    Return embedding for a single word.
    If gensim_model is None or word is unseen, return a zero vector.
    """
    if gensim_model is None:
        return np.zeros(embedding_dim, dtype=np.float32)
    if word in gensim_model.wv:
        return np.asarray(gensim_model.wv[word], dtype=np.float32)
    return np.zeros(embedding_dim, dtype=np.float32)


def words_to_embedding_matrix(words, embedding_dim, gensim_model=None):
    """
    Convert a list of words into an embedding matrix of shape
    (len(words), embedding_dim).
    """
    return np.array(
        [get_word_embedding(word, embedding_dim, gensim_model) for word in words],
        dtype=np.float32,
    )


def build_embedding_layer_from_gensim(vocab, embedding_dim, gensim_model=None, freeze=False):
    """
    Build a PyTorch embedding layer initialized from gensim vectors.
    """
    weight_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for word, idx in vocab.items():
        weight_matrix[idx] = get_word_embedding(word, embedding_dim, gensim_model)

    emb = nn.Embedding(len(vocab), embedding_dim)
    emb.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)
    emb.weight.requires_grad = not freeze
    return emb


def load_corpus(corpus_path: str | None = None, corpus_text: str | None = None):
    if corpus_text is not None:
        return build_vocab_from_corpus(corpus_text)
    if corpus_path is not None:
        with open(corpus_path, "r", encoding="utf-8") as f:
            return build_vocab_from_corpus(f.read())
    return build_vocab_from_corpus(DEFAULT_CORPUS)


def build_training_examples(tokens, vocab, sequence_length):
    """
    Build next-word prediction examples.

    Example:
        sequence_length = 3
        input  = [w0, w1, w2]
        target = w3
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    X, y = [], []
    for i in range(len(tokens) - sequence_length):
        input_words = tokens[i: i + sequence_length]
        target_word = tokens[i + sequence_length]
        X.append([vocab[w] for w in input_words])
        y.append(vocab[target_word])

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=500):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Create positional encoding matrix (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, embedding_dim)
        pe = pe.unsqueeze(0)

        # Register as buffer (not trainable)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embedding_dim)

        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Do not rename these. The tests expect these exact names.
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, embedding_dim)

        TODO:
        1. Compute Q, K, V using linear projections.
        2. Reshape/split into num_heads heads.
        3. Compute scaled dot-product attention per head.
        4. Concatenate heads back together.
        5. Apply output projection W_o.

        Return shape:
            (batch_size, sequence_length, embedding_dim)
        """
        batch_size, seq_len, embedding_dim = x.shape

        # 1. compute Q, K, V, batch size, sequence length, embedding dimension
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. reshape into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. scaled dot-product attention
        sqrt_d = math.sqrt(self.head_dim)

        scores = torch.matmul(Q, K.transpose(2, 3)) / sqrt_d
        softmax_res = torch.softmax(scores, dim=-1)

        # weight values by attention scores
        attn_output = torch.matmul(softmax_res, V)

        # 4. concatenate heads, reshape back to (batch_size, sequence_length, embedding_dim)
        # make contiguous for view after transpose
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)

        # 5. output projection W_o
        output = self.W_o(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()
        # Do not rename these. The tests expect block.attn.
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        TODO:
        1. Attention sublayer with residual connection and normalization.
        2. Feed-forward sublayer with residual connection and normalization.
        """
        # attention sublayer with residual connection and layer norm
        x = self.norm1(x + self.attn(x))

        # feed-forward sublayer with residual connection and layer norm
        x = self.norm2(x + self.ffn(x))

        return x


class SimplifiedTransformerLM(nn.Module):
    def __init__(self, vocab_size, sequence_length, embedding_dim, num_heads, num_blocks, ff_hidden_dim=128):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Do not rename these. The tests expect these exact names.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        """
        TODO:
        1. Convert token ids to embeddings via token_embedding.
        2. Add positional encodings via position_embedding.
        3. Pass through all transformer blocks sequentially.
        4. Take the representation at the last sequence position.
        5. Map to vocabulary logits using self.classifier.

        input_ids shape:
            (batch_size, sequence_length)
        return shape:
            (batch_size, vocab_size)
        """
        # 1. convert token ids to embeddings
        x = self.token_embedding(input_ids)
        # 2. add positional encodings
        x = self.position_embedding(x)

        # 3. pass through all transformer blocks
        for transformer_block in self.blocks:
            x = transformer_block(x)

        # 4. take the representation at the last sequence position
        last_pos = x[:, self.sequence_length - 1, :]

        # 5. map to vocabulary logits
        vocab_logits = self.classifier(last_pos)

        return vocab_logits


def train_transformer(
        corpus_path: str | None,
        num_heads: int,
        embedding_dim: int,
        sequence_length: int,
        num_blocks: int,
        num_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        corpus_text: str | None = None,
):
    """
    Train a simplified transformer language model.

    Returns
    -------
    model, vocab, idx_to_word, history
    """
    if embedding_dim % num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    set_seed(42)
    # TODO:
    # 1. Load corpus and build vocabulary.
    # 2. Build training examples.
    # 3. Create the model.
    # 4. Define optimizer and loss.
    # 5. Write the training loop.
    # 6. Return model, vocab, idx_to_word, history.

    # 1. load corpus and build vocabulary
    words, word_to_idx, idx_to_word = load_corpus(corpus_path, corpus_text)

    # 2. build training examples
    x, y = build_training_examples(words, word_to_idx, sequence_length)

    # 3. create the model
    vocab_size = len(word_to_idx)

    model = SimplifiedTransformerLM(vocab_size, sequence_length, embedding_dim, num_heads, num_blocks)

    # 4. define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # make dataloader for training loop
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # 5. Training Loop
    loss_history = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_batches = 0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            # forward pass
            output_preds = model(x_batch)

            loss = loss_func(output_preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

        avg_loss = epoch_loss / total_batches
        loss_history.append(avg_loss)

    # 6. Return model, vocab, idx_to_word, loss_history
    return model, word_to_idx, idx_to_word, loss_history


def _words_to_ids(words, vocab, required_length):
    if len(words) != required_length:
        raise ValueError(f"Expected {required_length} words, got {len(words)}")
    ids = [vocab.get(word.lower(), 0) for word in words]
    return torch.tensor([ids], dtype=torch.long)


def predict_next_word(model, vocab, idx_to_word, input_words):
    """
    Predict the next word from a sequence of input words.

    TODO:
    1. Convert input words to token id tensor.
    2. Run the model in eval mode without gradient computation.
    3. Select the highest-scoring output word.
    """
    # 1. convert input words to ids tensor
    input_ids = _words_to_ids(input_words, vocab, model.sequence_length)

    # 2. run model in eval mode
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    # 3. pick the word with the highest score, no softmax since it wants the highest score (temp = 1.0)
    predicted_idx = torch.argmax(output, dim=-1).item()

    return idx_to_word[predicted_idx]


def generate_text(model, vocab, idx_to_word, seed_words, num_words_to_generate):
    """
    Generate text autoregressively.

    TODO:
    1. Start with seed_words as the initial context.
    2. Repeatedly predict the next word using the last sequence_length words.
    3. Append each predicted word to the running sequence.
    4. Return the full generated sequence (seed + generated words).
    """
    # 1. initialize with seed words
    output_words = list(seed_words)

    # 2. repeatedly predict next word and append to output
    for i in range(num_words_to_generate):

        n = len(output_words)
        context_window = output_words[n - model.sequence_length:]
        next_word = predict_next_word(model, vocab, idx_to_word, context_window)

        # 3. append predicted word to output
        output_words.append(next_word)

    # 4. Return the full sequence
    return output_words


def _print_demo_predictions(model, vocab, idx_to_word, prompts: List[List[str]]) -> None:
    print("\nSample next-word predictions:")
    for prompt in prompts:
        try:
            prediction = predict_next_word(model, vocab, idx_to_word, prompt)
            print(f"  input: {' '.join(prompt):<35} -> predicted next word: {prediction}")
        except Exception as exc:
            print(f"  input: {' '.join(prompt)} -> error: {exc}")


def main() -> None:
    embedded_corpus_path = Path(__file__).with_name("embedded_corpus.txt")
    embedded_corpus_path.write_text(DEFAULT_CORPUS, encoding="utf-8")

    model, vocab, idx_to_word, history = train_transformer(
        corpus_path=None,
        corpus_text=DEFAULT_CORPUS,
        num_heads=2,
        embedding_dim=16,
        sequence_length=4,
        num_blocks=2,
        num_epochs=60,
        batch_size=8,
        learning_rate=1e-3,
    )

    print("Training completed.")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Final training loss: {history[-1]:.6f}")
    print(f"Embedded corpus saved to: {embedded_corpus_path}")

    prompts = [
        ["in", "a", "quiet", "town"],
        ["the", "students", "came", "from"],
        ["they", "all", "realized", "that"],
        ["most", "of", "all", "they"],
    ]
    _print_demo_predictions(model, vocab, idx_to_word, prompts)

    seed_words = ["in", "a", "quiet", "town"]
    generated = generate_text(model, vocab, idx_to_word, seed_words, num_words_to_generate=12)
    print("\nGenerated continuation:")
    print("  " + " ".join(generated))


if __name__ == "__main__":
    main()