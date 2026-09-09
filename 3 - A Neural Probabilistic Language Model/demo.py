"""
================================================================================
BENGIO ET AL. (2003) NEURAL PROBABILISTIC LANGUAGE MODEL - NATIVE NUMPY DEMO
================================================================================
Demonstrates:
  1. Mathematical Gradient Verification (`gradcheck`) against Finite Differences
  2. Dataset Preprocessing & Vocabulary Building
  3. Trigram Baseline Training (Interpolated Smoothing)
  4. Native Bengio NPLM Training (Mini-batch SGD with Weight Decay & Momentum)
  5. Perplexity Metric Comparison Table
  6. Distributed Word Embedding Cosine Similarity Inspection
  7. Interactive Text Generation / Completion
================================================================================
"""

import numpy as np
import time
from nplm_native import Vocabulary, NativeNPLM, NGramBaseline, prepare_dataset, tokenize
from train_native import train_nplm
from generate_native import get_most_similar_words, predict_next_word, generate_text
from test_native import run_all_tests


def run_demo():
    print("=" * 80)
    print("      A NEURAL PROBABILISTIC LANGUAGE MODEL (BENGIO ET AL., 2003)")
    print("                   PURE NATIVE NUMPY IMPLEMENTATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: GRADIENT CHECK (MATH VERIFICATION)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 1: VERIFYING MATHEMATICAL BACKPROPAGATION VIA FINITE DIFFERENCES")
    print("─" * 80)
    run_all_tests()

    # -------------------------------------------------------------------------
    # STEP 2: PREPARE TEXT DATASET & VOCABULARY
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 2: PREPARING CORPUS & VOCABULARY")
    print("─" * 80)

    corpus = [
        "the cat sits on the warm mat in the kitchen .",
        "the dog sits on the soft floor in the living room .",
        "the cat eats delicious fish and drinks milk .",
        "the dog eats crunchy food and drinks water .",
        "a woman drinks hot coffee in the morning .",
        "a man drinks hot tea in the afternoon .",
        "the boy plays soccer with his friends in the park .",
        "the girl reads an interesting book in the library .",
        "the cat sleeps peacefully on the cozy sofa .",
        "the dog barks loudly at the mailman in the morning .",
        "a man reads the daily newspaper in the garden .",
        "a woman prepares a tasty dinner in the kitchen .",
        "the boy rides a fast bicycle in the park .",
        "the girl writes a creative story in her notebook ."
    ]

    sentences = [tokenize(line) for line in corpus]
    vocab = Vocabulary()
    vocab.build_vocab(sentences, min_freq=1)

    print(f"Total Sentences  : {len(sentences)}")
    print(f"Vocabulary Size  : {len(vocab)} unique tokens")
    print(f"Sample Tokens    : {list(vocab.word2idx.keys())[:12]}...")

    context_len = 3  # (n-1) = 3 words to predict 4th word (4-gram setup)
    X_all, y_all = prepare_dataset(sentences, vocab, context_len=context_len)

    # Split into 80% Train, 20% Validation
    np.random.seed(42)
    indices = np.random.permutation(len(X_all))
    split_idx = int(0.8 * len(X_all))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    print(f"Total N-gram Samples : {len(X_all)}")
    print(f"Training Samples     : {len(X_train)}")
    print(f"Validation Samples   : {len(X_val)}")

    # -------------------------------------------------------------------------
    # STEP 3: TRAIN TRIGRAM BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 3: EVALUATING TRADITIONAL N-GRAM BASELINE (INTERPOLATED TRIGRAM)")
    print("─" * 80)

    train_sentences = [sentences[i] for i in range(len(sentences)) if i % 5 != 0]
    val_sentences = [sentences[i] for i in range(len(sentences)) if i % 5 == 0]

    ngram_model = NGramBaseline(n=context_len + 1, smoothing="interpolation", lambdas=(0.5, 0.35, 0.15))
    ngram_model.fit(train_sentences)

    ngram_train_ppl = ngram_model.compute_perplexity(X_train, y_train, vocab)
    ngram_val_ppl = ngram_model.compute_perplexity(X_val, y_val, vocab)

    print(f"  Trigram Training Perplexity   : {ngram_train_ppl:8.2f}")
    print(f"  Trigram Validation Perplexity : {ngram_val_ppl:8.2f}")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN BENGIO ET AL. (2003) NATIVE NPLM
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 4: TRAINING BENGIO ET AL. (2003) NATIVE NPLM MODEL")
    print("─" * 80)

    embedding_dim = 24
    hidden_dim = 48
    nplm_model = NativeNPLM(
        vocab_size=len(vocab),
        context_len=context_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        direct_connections=True,
        seed=42
    )

    print(f"Model Architecture Settings:")
    print(f"  - Embedding dimension (m)       : {embedding_dim}")
    print(f"  - Hidden layer neurons (h)      : {hidden_dim}")
    print(f"  - Context length (n-1)          : {context_len}")
    print(f"  - Direct Input-to-Output Matrix U: Enabled (True)")

    history = train_nplm(
        model=nplm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=35,
        batch_size=16,
        lr=0.07,
        lr_decay=0.96,
        momentum=0.9,
        l2_reg=1e-4,
        verbose=True
    )

    nplm_train_ppl = history['train_ppl'][-1]
    nplm_val_ppl = history['val_ppl'][-1]

    # -------------------------------------------------------------------------
    # STEP 5: PERPLEXITY COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: FINAL MODEL PERFORMANCE & PERPLEXITY COMPARISON")
    print("=" * 80)
    print(f"{'Model Architecture':<40} | {'Train PPL':<15} | {'Val PPL':<15}")
    print("-" * 78)
    print(f"{'Traditional Trigram (Interpolation)':<40} | {ngram_train_ppl:<15.2f} | {ngram_val_ppl:<15.2f}")
    print(f"{'Bengio (2003) Native NPLM Model':<40} | {nplm_train_ppl:<15.2f} | {nplm_val_ppl:<15.2f}")
    print("-" * 78)
    improvement = ((ngram_val_ppl - nplm_val_ppl) / ngram_val_ppl) * 100
    print(f"Relative Perplexity Improvement over N-gram: {improvement:+.2f}% 🎉")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 6: DISTRIBUTED WORD EMBEDDINGS (COSINE SIMILARITY)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 6: INSPECTING LEARNED DISTRIBUTED WORD EMBEDDINGS (MATRIX C)")
    print("─" * 80)
    query_words = ["cat", "dog", "coffee", "kitchen"]

    for word in query_words:
        if word in vocab.word2idx:
            sims = get_most_similar_words(nplm_model, vocab, word, top_k=4)
            sim_str = ", ".join([f"{w} ({sim:.3f})" for w, sim in sims])
            print(f"  Most similar to '{word:8s}': {sim_str}")

    # -------------------------------------------------------------------------
    # STEP 7: TEXT GENERATION & NEXT-WORD PREDICTION
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("STEP 7: NEXT-WORD PREDICTION & AUTOREGRESSIVE TEXT GENERATION")
    print("─" * 80)

    prompts = [
        ["the", "cat", "sits"],
        ["a", "woman", "drinks"],
        ["the", "dog", "eats"]
    ]

    for prompt in prompts:
        candidates = predict_next_word(nplm_model, vocab, prompt, top_k=3, temperature=1.0)
        cand_str = ", ".join([f"'{w}' ({p * 100:.1f}%)" for w, p in candidates])
        print(f"  Prompt: {' '.join(prompt):<22} -> Next Word Top Predictions: {cand_str}")

    print("\n  Sample Autoregressive Text Generation:")
    for prompt in prompts:
        generated = generate_text(nplm_model, vocab, prompt, max_tokens=8, temperature=0.7, top_k=3)
        print(f"    Prompt: {' '.join(prompt):<20} -> Output: '{generated}'")

    print("\n" + "=" * 80)
    print("DEMO EXECUTED SUCCESSFULLY! BENGIO ET AL. (2003) FULLY VERIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
