"""
Text Generation, Next-Word Prediction, and Word Embedding Cosine Similarity Lookup
for Native Bengio et al. (2003) Neural Language Model.
"""

import numpy as np
from nplm_native import Vocabulary, NativeNPLM, tokenize


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def get_most_similar_words(model, vocab, target_word, top_k=5):
    """
    Find top_k words with highest cosine similarity to target_word in embedding matrix C.
    """
    if target_word not in vocab.word2idx:
        print(f"Word '{target_word}' not found in vocabulary!")
        return []

    target_idx = vocab.word2idx[target_word]
    target_vec = model.C[target_idx]

    similarities = []
    for word, idx in vocab.word2idx.items():
        if word == target_word or word in [vocab.pad_token, vocab.unk_token, vocab.sos_token, vocab.eos_token]:
            continue
        vec = model.C[idx]
        sim = cosine_similarity(target_vec, vec)
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def predict_next_word(model, vocab, prompt_tokens, top_k=5, temperature=1.0):
    """
    Given prompt tokens, predict probability distribution over next word candidates.
    """
    context_len = model.context_len
    # Pad or truncate prompt to context_len
    if len(prompt_tokens) < context_len:
        padded_tokens = [vocab.sos_token] * (context_len - len(prompt_tokens)) + prompt_tokens
    else:
        padded_tokens = prompt_tokens[-context_len:]

    context_indices = np.array([vocab.encode(padded_tokens)], dtype=np.int32)
    
    # Forward pass to get logits / scores
    probs, _ = model.forward(context_indices)
    probs = probs[0]  # Shape: (vocab_size,)

    # Apply temperature
    if temperature != 1.0:
        log_probs = np.log(probs + 1e-15) / temperature
        exp_probs = np.exp(log_probs - np.max(log_probs))
        probs = exp_probs / np.sum(exp_probs)

    top_indices = np.argsort(probs)[::-1][:top_k]
    candidates = []
    for idx in top_indices:
        word = vocab.idx2word[idx]
        candidates.append((word, probs[idx]))

    return candidates


def generate_text(model, vocab, prompt_tokens, max_tokens=10, temperature=0.8, top_k=3):
    """
    Autoregressively generate next words given an initial prompt.
    """
    current_tokens = list(prompt_tokens)

    for _ in range(max_tokens):
        candidates = predict_next_word(model, vocab, current_tokens, top_k=top_k, temperature=temperature)
        words = [c[0] for c in candidates]
        p_vals = np.array([c[1] for c in candidates])
        p_vals = p_vals / np.sum(p_vals)  # Re-normalize

        # Sample next word according to top_k probability distribution
        chosen_word = np.random.choice(words, p=p_vals)
        if chosen_word == vocab.eos_token:
            break
        current_tokens.append(chosen_word)

    return " ".join(current_tokens)
