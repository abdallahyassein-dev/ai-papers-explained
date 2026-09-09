"""
Training Pipeline for Native NumPy Bengio et al. (2003) Neural Language Model.
Includes mini-batch Stochastic Gradient Descent (SGD) with momentum, weight decay, learning rate decay,
and validation evaluation against N-gram baseline.
"""

import numpy as np
import math
import time
from nplm_native import Vocabulary, NativeNPLM, NGramBaseline, prepare_dataset, tokenize


def train_nplm(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, 
               lr=0.05, lr_decay=0.95, momentum=0.9, l2_reg=1e-4, verbose=True):
    """
    Train Native NPLM using mini-batch SGD with momentum and weight decay.
    """
    num_samples = len(X_train)
    history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}

    if verbose:
        print(f"Starting Training: {num_samples} training samples, {len(X_val)} validation samples")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, Initial LR: {lr}, Momentum: {momentum}, L2 Reg: {l2_reg}")
        print("-" * 75)

    current_lr = lr

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Shuffle training data at each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = math.ceil(num_samples / batch_size)

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # 1. Forward Pass
            probs, loss = model.forward(X_batch, y_target=y_batch, l2_reg=l2_reg)
            epoch_loss += loss * (end_idx - start_idx)

            # 2. Backward Pass
            model.backward(l2_reg=l2_reg)

            # 3. Parameter Update
            model.step(lr=current_lr, momentum=momentum)

        # Average training loss for the epoch
        avg_train_loss = epoch_loss / num_samples
        train_ppl = model.compute_perplexity(X_train, y_train)
        val_ppl = model.compute_perplexity(X_val, y_val)
        _, val_loss = model.forward(X_val, y_target=y_val, l2_reg=l2_reg)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        elapsed = time.time() - start_time

        if verbose:
            print(f"Epoch {epoch:2d}/{epochs:2d} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:6.2f} | "
                  f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | "
                  f"LR: {current_lr:.5f} | Time: {elapsed:.2f}s")

        # Learning rate decay
        current_lr *= lr_decay

    if verbose:
        print("-" * 75)
        print("Training Complete! Finished all epochs.")

    return history


def main():
    # Sample training corpus illustrating natural language structure
    sample_text = """
    the cat sits on the mat in the living room .
    the dog runs in the park on a sunny day .
    the cat eats fish and drinks milk .
    the dog barks at the mailman in the morning .
    a woman drinks coffee in the kitchen .
    a man drinks tea in the garden .
    the boy plays soccer in the park with his friends .
    the girl reads a book in the library .
    the cat sleeps on the warm sofa .
    the dog chases the cat around the house .
    a man reads the news in the morning .
    a woman cooks dinner in the kitchen .
    """

    # Tokenize and split sentences
    sentences = [tokenize(line) for line in sample_text.strip().split('\n') if line.strip()]

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(sentences, min_freq=1)

    print(f"Vocabulary size: {len(vocab)} words")

    # Prepare datasets with context_len=3 (predict next word using 3 previous words)
    context_len = 3
    X_data, y_data = prepare_dataset(sentences, vocab, context_len=context_len)

    # Train/Validation split (80% train, 20% val)
    split_idx = int(0.8 * len(X_data))
    X_train, y_train = X_data[:split_idx], y_data[:split_idx]
    X_val, y_val = X_data[split_idx:], y_data[split_idx:]

    print(f"Dataset samples - Train: {len(X_train)}, Val: {len(X_val)}")

    # 1. Baseline N-gram Model
    print("\n" + "=" * 50)
    print("1. TRAINING N-GRAM BASELINE (TRIGRAM WITH INTERPOLATION)")
    print("=" * 50)
    ngram_model = NGramBaseline(n=context_len + 1, smoothing="interpolation")
    ngram_model.fit(sentences[:int(0.8 * len(sentences))])
    ngram_ppl = ngram_model.compute_perplexity(X_val, y_val, vocab)
    print(f"N-gram Baseline Validation Perplexity: {ngram_ppl:.2f}")

    # 2. Bengio et al. (2003) Native NPLM
    print("\n" + "=" * 50)
    print("2. TRAINING BENGIO ET AL. (2003) NATIVE NPLM MODEL")
    print("=" * 50)
    embedding_dim = 16
    hidden_dim = 32
    nplm_model = NativeNPLM(
        vocab_size=len(vocab),
        context_len=context_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        direct_connections=True,
        seed=42
    )

    history = train_nplm(
        model=nplm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=30,
        batch_size=16,
        lr=0.08,
        lr_decay=0.96,
        momentum=0.9,
        l2_reg=1e-4,
        verbose=True
    )

    final_val_ppl = history['val_ppl'][-1]
    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON SUMMARY (PERPLEXITY - LOWER IS BETTER):")
    print(f"  Trigram Baseline (Interpolated): {ngram_ppl:8.2f}")
    print(f"  Bengio (2003) Native NPLM Model:  {final_val_ppl:8.2f}")
    improvement = ((ngram_ppl - final_val_ppl) / ngram_ppl) * 100
    print(f"  Relative Perplexity Improvement: {improvement:+6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
