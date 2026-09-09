"""
Bengio et al. (2003) - A Neural Probabilistic Language Model (NPLM)
Pure Native Python & NumPy Implementation (No PyTorch/Autograd Frameworks)

Equations implemented:
  x = [C(w_{t-n+1}) ; C(w_{t-n+2}) ; ... ; C(w_{t-1})]   (Concatenation of context word embeddings)
  a = d + H * x                                          (Hidden layer pre-activation)
  hidden = tanh(a)                                       (Hidden activation)
  y = b + W * hidden + U * x                             (Output score / logits)
  P(w_t | context) = softmax(y)                          (Probability distribution)
  Loss = -1/T * sum log P(w_t) + lambda/2 * ||theta||^2  (Negative Log-Likelihood + L2 Regularization)
"""

import numpy as np
import re
import math
from collections import Counter, defaultdict


class Vocabulary:
    """
    Vocabulary manager for mapping tokens to unique integer indices and back.
    Includes special tokens: <pad>, <unk>, <sos>, <eos>.
    """
    def __init__(self, pad_token="<pad>", unk_token="<unk>", sos_token="<sos>", eos_token="<eos>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.word2idx = {}
        self.idx2word = {}
        
        # Add special tokens first
        for tok in [pad_token, unk_token, sos_token, eos_token]:
            self.add_token(tok)

    def add_token(self, token):
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            return idx
        return self.word2idx[token]

    def build_vocab(self, sentences, min_freq=1):
        """Build vocabulary from a list of tokenized sentences."""
        counter = Counter()
        for sent in sentences:
            counter.update(sent)
        
        for token, freq in counter.most_common():
            if freq >= min_freq:
                self.add_token(token)

    def encode(self, tokens):
        """Convert list of tokens to list of indices."""
        return [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in tokens]

    def decode(self, indices):
        """Convert list of indices to list of tokens."""
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

    def __len__(self):
        return len(self.word2idx)


def tokenize(text):
    """Simple clean word tokenizer."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens


def prepare_dataset(sentences, vocab, context_len=3):
    """
    Given tokenized sentences, create (context, target) training samples.
    context_len: number of previous words (n-1) used to predict next word.
    """
    X_samples = []
    y_samples = []

    for sent in sentences:
        # Prepend <sos> context_len times, append <eos>
        padded = [vocab.sos_token] * context_len + sent + [vocab.eos_token]
        indices = vocab.encode(padded)
        
        for i in range(len(indices) - context_len):
            context = indices[i : i + context_len]
            target = indices[i + context_len]
            X_samples.append(context)
            y_samples.append(target)

    return np.array(X_samples, dtype=np.int32), np.array(y_samples, dtype=np.int32)


class NGramBaseline:
    """
    Traditional N-gram Language Model baseline (Trigram / N-gram).
    Implements Laplace (Add-k) and Linear Interpolation smoothing for comparison.
    """
    def __init__(self, n=3, smoothing="interpolation", add_k=0.1, lambdas=(0.6, 0.3, 0.1)):
        self.n = n
        self.smoothing = smoothing
        self.add_k = add_k
        self.lambdas = lambdas  # weights for (trigram, bigram, unigram)
        
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.unigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()

    def fit(self, sentences):
        """Train the N-gram counter on tokenized sentences."""
        for sent in sentences:
            padded = ["<sos>"] * (self.n - 1) + sent + ["<eos>"]
            for i in range(len(padded)):
                word = padded[i]
                self.vocab.add(word)
                self.unigram_counts[word] += 1
                self.total_words += 1

                if i >= self.n - 1:
                    context = tuple(padded[i - self.n + 1 : i])
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1

    def get_word_prob(self, context, word):
        """Calculate conditional probability P(word | context)."""
        vocab_size = len(self.vocab)
        context = tuple(context[-(self.n - 1):])

        if self.smoothing == "laplace":
            count_ngram = self.ngram_counts[context][word]
            count_ctx = self.context_counts[context]
            return (count_ngram + self.add_k) / (count_ctx + self.add_k * vocab_size)

        elif self.smoothing == "interpolation":
            # Trigram component
            count_tri = self.ngram_counts[context][word]
            count_tri_ctx = self.context_counts[context]
            p_tri = count_tri / count_tri_ctx if count_tri_ctx > 0 else 1.0 / vocab_size

            # Bigram component
            bigram_ctx = tuple(context[-1:])
            count_bi = sum(self.ngram_counts[ctx][word] for ctx in self.ngram_counts if len(ctx) >= 1 and ctx[-1] == context[-1])
            count_bi_ctx = sum(self.context_counts[ctx] for ctx in self.context_counts if len(ctx) >= 1 and ctx[-1] == context[-1])
            p_bi = count_bi / count_bi_ctx if count_bi_ctx > 0 else 1.0 / vocab_size

            # Unigram component
            p_uni = self.unigram_counts[word] / self.total_words if self.total_words > 0 else 1.0 / vocab_size

            l1, l2, l3 = self.lambdas
            return l1 * p_tri + l2 * p_bi + l3 * p_uni
        
        else:
            count_ngram = self.ngram_counts[context][word]
            count_ctx = self.context_counts[context]
            return count_ngram / count_ctx if count_ctx > 0 else 1.0 / vocab_size

    def compute_perplexity(self, X_samples, y_samples, vocab):
        """Calculate Perplexity metric on evaluation dataset."""
        log_prob_sum = 0.0
        total_tokens = len(y_samples)

        for context_indices, target_idx in zip(X_samples, y_samples):
            context_words = vocab.decode(context_indices)
            target_word = vocab.idx2word.get(target_idx, vocab.unk_token)
            p = self.get_word_prob(context_words, target_word)
            p = max(p, 1e-12)
            log_prob_sum += math.log(p)

        nll = -log_prob_sum / total_tokens
        perplexity = math.exp(nll)
        return perplexity


class NativeNPLM:
    """
    Bengio et al. (2003) Neural Probabilistic Language Model implemented natively in NumPy.
    
    Parameters:
      vocab_size (int): Total number of words in vocabulary |V|
      context_len (int): Context length n-1 (e.g. 3 for 4-gram)
      embedding_dim (int): Vector dimension m for word embeddings
      hidden_dim (int): Number of hidden units h
      direct_connections (bool): Whether to include direct matrix U from input to output
      seed (int): Random seed for reproducibility
    """
    def __init__(self, vocab_size, context_len=3, embedding_dim=30, hidden_dim=50, direct_connections=True, seed=42):
        np.random.seed(seed)

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.m = embedding_dim
        self.h = hidden_dim
        self.input_dim = context_len * embedding_dim
        self.direct_connections = direct_connections

        # 1. Embedding Matrix C: |V| x m
        self.C = np.random.randn(vocab_size, self.m) * (1.0 / np.sqrt(self.m))

        # 2. Hidden Layer Weights H: h x (context_len * m), Bias d: h
        self.H = np.random.randn(self.h, self.input_dim) * (1.0 / np.sqrt(self.input_dim))
        self.d = np.zeros(self.h)

        # 3. Output Layer Weights W: |V| x h, Bias b: |V|
        self.W = np.random.randn(vocab_size, self.h) * (1.0 / np.sqrt(self.h))
        self.b = np.zeros(vocab_size)

        # 4. Direct Connection Matrix U: |V| x (context_len * m)
        if self.direct_connections:
            self.U = np.random.randn(vocab_size, self.input_dim) * (1.0 / np.sqrt(self.input_dim))
        else:
            self.U = None

        # Gradients cache
        self.dC = np.zeros_like(self.C)
        self.dH = np.zeros_like(self.H)
        self.dd = np.zeros_like(self.d)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dU = np.zeros_like(self.U) if self.direct_connections else None

        # Momentum velocities cache
        self.vC = np.zeros_like(self.C)
        self.vH = np.zeros_like(self.H)
        self.vd = np.zeros_like(self.d)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.vU = np.zeros_like(self.U) if self.direct_connections else None

        # Forward pass cache for backpropagation
        self.cache = {}

    def forward(self, X, y_target=None, l2_reg=0.0):
        """
        Forward pass.
        Inputs:
          X: Integer array of shape (batch_size, context_len) containing context word indices
          y_target: Integer array of shape (batch_size,) containing target word indices
          l2_reg: Weight decay hyperparameter lambda
        
        Returns:
          probs: Probabilities of shape (batch_size, vocab_size)
          loss: NLL loss + L2 regularization penalty (scalar)
        """
        batch_size = X.shape[0]

        # 1. Embedding lookup and concatenation: x = [C(w_1); C(w_2); ...; C(w_{n-1})]
        # Shape: (batch_size, context_len, m) -> reshape to (batch_size, context_len * m)
        embeddings = self.C[X]  # (B, context_len, m)
        x = embeddings.reshape(batch_size, self.input_dim)  # (B, input_dim)

        # 2. Hidden layer pre-activation & tanh: a = d + H * x, hidden = tanh(a)
        a = x.dot(self.H.T) + self.d  # (B, h)
        hidden = np.tanh(a)  # (B, h)

        # 3. Output logits: y = b + W * hidden + U * x
        y = hidden.dot(self.W.T) + self.b  # (B, vocab_size)
        if self.direct_connections:
            y += x.dot(self.U.T)  # (B, vocab_size)

        # 4. Numerically stable Softmax: P = exp(y - max(y)) / sum(exp(y - max(y)))
        y_max = np.max(y, axis=1, keepdims=True)
        exp_y = np.exp(y - y_max)
        probs = exp_y / np.sum(exp_y, axis=1, keepdims=True)  # (B, vocab_size)

        # Cache variables needed for backward pass
        self.cache = {
            'X': X,
            'x': x,
            'a': a,
            'hidden': hidden,
            'probs': probs,
            'y_target': y_target
        }

        # Calculate loss if target provided
        loss = None
        if y_target is not None:
            # Negative Log-Likelihood (NLL)
            correct_log_probs = np.log(probs[np.arange(batch_size), y_target] + 1e-15)
            nll_loss = -np.mean(correct_log_probs)

            # L2 Regularization (Weight Decay) penalty on weights
            l2_loss = 0.5 * l2_reg * (
                np.sum(self.W ** 2) +
                np.sum(self.H ** 2) +
                np.sum(self.C ** 2) +
                (np.sum(self.U ** 2) if self.direct_connections else 0.0)
            )
            loss = nll_loss + l2_loss

        return probs, loss

    def backward(self, l2_reg=0.0):
        """
        Analytical Backward Pass (Manual Backpropagation).
        Computes exact derivatives for C, H, d, W, b, U.
        """
        X = self.cache['X']
        x = self.cache['x']
        hidden = self.cache['hidden']
        probs = self.cache['probs']
        y_target = self.cache['y_target']
        batch_size = X.shape[0]

        # 1. Gradient of NLL loss w.r.t logits y: dScores = (P - 1_y) / B
        dScores = probs.copy()
        dScores[np.arange(batch_size), y_target] -= 1.0
        dScores /= batch_size  # (B, vocab_size)

        # 2. Gradients for Output layer parameters: W, b, and U
        self.dW = dScores.T.dot(hidden) + l2_reg * self.W  # (vocab_size, h)
        self.db = np.sum(dScores, axis=0)  # (vocab_size,)

        if self.direct_connections:
            self.dU = dScores.T.dot(x) + l2_reg * self.U  # (vocab_size, input_dim)

        # 3. Backprop to hidden layer: da_hidden = dScores * W
        da_hidden = dScores.dot(self.W)  # (B, h)

        # 4. Backprop through tanh: d/da tanh(a) = 1 - tanh(a)^2 = 1 - hidden^2
        da = da_hidden * (1.0 - hidden ** 2)  # (B, h)

        # 5. Gradients for Hidden layer parameters: H, d
        self.dH = da.T.dot(x) + l2_reg * self.H  # (h, input_dim)
        self.dd = np.sum(da, axis=0)  # (h,)

        # 6. Backprop to concatenated embeddings x: dx = da * H (+ dScores * U if direct)
        dx = da.dot(self.H)  # (B, input_dim)
        if self.direct_connections:
            dx += dScores.dot(self.U)  # (B, input_dim)

        # 7. Gradient for Embedding Matrix C
        # Reset embedding gradients
        self.dC.fill(0.0)

        # Reshape dx to (batch_size, context_len, m)
        dx_reshaped = dx.reshape(batch_size, self.context_len, self.m)

        # Scatter-add gradients back to corresponding word indices in C
        for i in range(batch_size):
            for c_pos in range(self.context_len):
                word_idx = X[i, c_pos]
                self.dC[word_idx] += dx_reshaped[i, c_pos]

        # Add weight decay for embedding matrix (optional/standard)
        if l2_reg > 0:
            self.dC += l2_reg * self.C

    def step(self, lr=0.01, momentum=0.9):
        """
        Update parameters using Stochastic Gradient Descent (SGD) with momentum.
        """
        # Update W, b
        self.vW = momentum * self.vW + self.dW
        self.W -= lr * self.vW

        self.vb = momentum * self.vb + self.db
        self.b -= lr * self.vb

        # Update H, d
        self.vH = momentum * self.vH + self.dH
        self.H -= lr * self.vH

        self.vd = momentum * self.vd + self.dd
        self.d -= lr * self.vd

        # Update C
        self.vC = momentum * self.vC + self.dC
        self.C -= lr * self.vC

        # Update U if direct connections enabled
        if self.direct_connections:
            self.vU = momentum * self.vU + self.dU
            self.U -= lr * self.vU

    def compute_perplexity(self, X_eval, y_eval, batch_size=128):
        """Compute evaluation Perplexity score."""
        num_samples = len(y_eval)
        total_nll = 0.0

        for i in range(0, num_samples, batch_size):
            X_batch = X_eval[i : i + batch_size]
            y_batch = y_eval[i : i + batch_size]
            probs, _ = self.forward(X_batch)
            
            b_size = len(y_batch)
            correct_probs = probs[np.arange(b_size), y_batch]
            correct_probs = np.maximum(correct_probs, 1e-15)
            total_nll += -np.sum(np.log(correct_probs))

        avg_nll = total_nll / num_samples
        perplexity = np.exp(avg_nll)
        return perplexity
