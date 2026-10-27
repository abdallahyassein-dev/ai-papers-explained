# 📄 Attention Is All You Need: The Transformer, Explained from Scratch

> **Paper Title:** Attention Is All You Need  
> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
> **Institutions:** Google Brain, Google Research, University of Toronto  
> **Conference:** 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA  
> **ArXiv ID:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
> **Official PDF:** [`1706.03762v7.pdf`](1706.03762v7.pdf)  
> **Official Code:** [tensor2tensor](https://github.com/tensorflow/tensor2tensor)  
> **Milestone:** Introduced the **Transformer**, the first sequence transduction model built **entirely on attention**, with no recurrence and no convolution. It is the ancestor of BERT, GPT, T5, ViT, and practically every modern Large Language Model.

---

## 📑 Table of Contents

1. [Executive Summary: What Changed and Why It Matters](#1-executive-summary-what-changed-and-why-it-matters)
2. [Glossary of Terms (Beginner's Reference)](#2-glossary-of-terms-beginners-reference)
3. [The Problem: Sequence Transduction Before the Transformer](#3-the-problem-sequence-transduction-before-the-transformer)
   - [3.1 What Is Sequence Transduction?](#31-what-is-sequence-transduction)
   - [3.2 Recurrent Neural Networks and the Sequential Bottleneck](#32-recurrent-neural-networks-and-the-sequential-bottleneck)
   - [3.3 Attention as an Add-on to RNNs](#33-attention-as-an-add-on-to-rnns)
   - [3.4 Convolutional Alternatives: Extended Neural GPU, ByteNet, ConvS2S](#34-convolutional-alternatives-extended-neural-gpu-bytenet-convs2s)
   - [3.5 Self-Attention Before the Transformer](#35-self-attention-before-the-transformer)
4. [The Big Picture: Transformer Architecture Overview](#4-the-big-picture-transformer-architecture-overview)
5. [Step 0: From Words to Vectors (Tokenization & Embeddings)](#5-step-0-from-words-to-vectors-tokenization--embeddings)
6. [Positional Encoding: Giving the Model a Sense of Order](#6-positional-encoding-giving-the-model-a-sense-of-order)
7. [Attention: The Core Idea](#7-attention-the-core-idea)
   - [7.1 Queries, Keys and Values (The Library Analogy)](#71-queries-keys-and-values-the-library-analogy)
   - [7.2 Scaled Dot-Product Attention](#72-scaled-dot-product-attention)
   - [7.3 Full Numerical Walkthrough of Scaled Dot-Product Attention](#73-full-numerical-walkthrough-of-scaled-dot-product-attention)
   - [7.4 Why Divide by √dₖ? (The Scaling Factor)](#74-why-divide-by-dk-the-scaling-factor)
   - [7.5 Dot-Product vs. Additive Attention](#75-dot-product-vs-additive-attention)
8. [Multi-Head Attention](#8-multi-head-attention)
   - [8.1 Why Multiple Heads?](#81-why-multiple-heads)
   - [8.2 The Math](#82-the-math)
   - [8.3 Numerical Walkthrough (2 Heads)](#83-numerical-walkthrough-2-heads)
   - [8.4 Tensor Shapes in the Real Model](#84-tensor-shapes-in-the-real-model)
   - [8.5 Computational Cost of Multi-Head vs. Single-Head](#85-computational-cost-of-multi-head-vs-single-head)
9. [The Three Ways Attention Is Used in the Transformer](#9-the-three-ways-attention-is-used-in-the-transformer)
   - [9.1 Encoder Self-Attention](#91-encoder-self-attention)
   - [9.2 Masked Decoder Self-Attention (with Numbers)](#92-masked-decoder-self-attention-with-numbers)
   - [9.3 Encoder–Decoder (Cross) Attention](#93-encoderdecoder-cross-attention)
10. [Position-wise Feed-Forward Networks](#10-position-wise-feed-forward-networks)
11. [Residual Connections & Layer Normalization](#11-residual-connections--layer-normalization)
12. [The Encoder Stack in Detail](#12-the-encoder-stack-in-detail)
13. [The Decoder Stack in Detail](#13-the-decoder-stack-in-detail)
14. [The Output Layer: Linear + Softmax & Weight Sharing](#14-the-output-layer-linear--softmax--weight-sharing)
15. [Counting Parameters: Where Do 65M and 213M Come From?](#15-counting-parameters-where-do-65m-and-213m-come-from)
16. [Why Self-Attention? (Complexity, Parallelism, Path Length)](#16-why-self-attention-complexity-parallelism-path-length)
17. [Training Setup](#17-training-setup)
   - [17.1 Data and Batching](#171-data-and-batching)
   - [17.2 Hardware and Schedule](#172-hardware-and-schedule)
   - [17.3 The Adam Optimizer & the Warmup Learning-Rate Schedule](#173-the-adam-optimizer--the-warmup-learning-rate-schedule)
   - [17.4 Regularization: Residual Dropout](#174-regularization-residual-dropout)
   - [17.5 Regularization: Label Smoothing](#175-regularization-label-smoothing)
18. [Inference: Autoregressive Decoding, Beam Search, Checkpoint Averaging](#18-inference-autoregressive-decoding-beam-search-checkpoint-averaging)
19. [Results: Machine Translation](#19-results-machine-translation)
20. [Model Variations (Ablation Study, Table 3)](#20-model-variations-ablation-study-table-3)
21. [Generalization: English Constituency Parsing](#21-generalization-english-constituency-parsing)
22. [Attention Visualizations (Interpretability)](#22-attention-visualizations-interpretability)
23. [End-to-End Worked Example: Translating "I love cats"](#23-end-to-end-worked-example-translating-i-love-cats)
24. [Author Contributions](#24-author-contributions)
25. [Limitations, Subtleties & Beginner FAQ](#25-limitations-subtleties--beginner-faq)
26. [Summary & Lasting Legacy](#26-summary--lasting-legacy)

---

## 1. Executive Summary: What Changed and Why It Matters

In 2017, the dominant models for tasks like machine translation were **recurrent neural networks (RNNs)**, especially LSTMs and GRUs, arranged in an **encoder–decoder** structure and usually helped by an **attention mechanism**. These models read a sentence **one word at a time**, which made them slow to train and bad at linking words that are far apart.

The paper asks a bold question:

> *What if we throw away recurrence and convolutions completely and use **only attention**?*

The answer is the **Transformer**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         THE TRANSFORMER IN ONE BOX                        │
│                                                                           │
│   RNN encoder–decoder + attention   ──►   Attention ONLY (no recurrence)  │
│                                                                           │
│   • Reads words one by one         ──►   Reads all words in parallel      │
│   • Path between word 1 and word n ──►   Path length = 1 (direct link)    │
│     grows with n                                                          │
│   • Days/weeks of training         ──►   12 hours on 8 GPUs (base model)  │
│                                                                           │
│   WMT'14 English→German:  28.4 BLEU  (new SOTA, +2 BLEU over ensembles)   │
│   WMT'14 English→French:  41.8 BLEU  (new single-model SOTA, 3.5 days)    │
└───────────────────────────────────────────────────────────────────────────┘
```

### The Key Contributions

1. **Scaled Dot-Product Attention** — a fast, simple way to let every word "look at" every other word, with a scaling factor $1/\sqrt{d_k}$ that keeps training stable.
2. **Multi-Head Attention** — running several attention operations in parallel, each in a smaller subspace, so the model can capture different relationships at the same time (syntax, coreference, etc.).
3. **Sinusoidal Positional Encoding** — since there is no recurrence, word order is injected by adding fixed sine/cosine signals to the embeddings.
4. **A fully attention-based encoder–decoder** — 6 encoder layers + 6 decoder layers, each with attention + feed-forward sub-layers, residual connections and layer normalization.
5. **State-of-the-art results at a fraction of the cost** — better BLEU than all previous models (including ensembles) with far less training compute, and good generalization to English constituency parsing.

---

## 2. Glossary of Terms (Beginner's Reference)

| Term | Symbol | Technical Meaning | Intuitive Analogy |
|:---|:---:|:---|:---|
| **Token** | — | A unit of text (word or sub-word piece) after tokenization. | A Lego brick of a sentence. |
| **Sequence transduction** | — | Mapping an input sequence to an output sequence (e.g., English → German). | A translator converting one string of words into another. |
| **Embedding** | $d_{model}$ | A learned vector representing a token. In the paper $d_{model}=512$. | A 512-number "ID card" describing a word's meaning. |
| **Encoder** | — | Network that reads the input and produces continuous representations $\mathbf{z}$. | The reader who fully understands the source sentence. |
| **Decoder** | — | Network that generates the output one token at a time, using $\mathbf{z}$. | The writer who produces the translation word by word. |
| **Auto-regressive** | — | Each generated token is fed back as input to generate the next. | Writing a sentence where every new word depends on what you already wrote. |
| **Attention** | — | Weighted average of values, where weights come from query–key similarity. | Deciding how much to "listen" to each word. |
| **Self-attention** | — | Attention where queries, keys and values all come from the same sequence. | Words in a sentence talking to each other. |
| **Query / Key / Value** | $Q, K, V$ | Three projections of token vectors used to compute attention. | Search text / book labels / book contents. |
| **Head** | $h$ | One independent attention computation. Paper uses $h=8$. | One "expert" looking for one type of relation. |
| **$d_k$, $d_v$** | — | Dimensions of keys/queries and values per head ($64$ in the paper). | The size of each expert's notebook. |
| **Feed-forward network (FFN)** | $d_{ff}$ | Two linear layers with ReLU, applied to each position separately. $d_{ff}=2048$. | Each word "thinking privately" after the group discussion. |
| **Residual connection** | — | Adding a sub-layer's input to its output: $x + \text{Sublayer}(x)$. | A shortcut road so information/gradients don't get lost. |
| **Layer normalization** | LayerNorm | Normalizing each vector to mean 0 and variance 1 (then rescaling). | Re-centering a scale before each measurement. |
| **Positional encoding** | PE | Vector added to embeddings to encode word position. | Seat numbers in a theater. |
| **Masking** | — | Setting some attention scores to $-\infty$ so they get weight 0. | Covering future words with your hand so you can't cheat. |
| **Softmax** | — | $\text{softmax}(x_i) = e^{x_i}/\sum_j e^{x_j}$, turns scores into probabilities. | Turning raw votes into percentages that sum to 100%. |
| **BLEU** | — | Machine-translation metric based on n-gram overlap with references (0–100). | A score of how close the translation is to a human one. |
| **Perplexity (PPL)** | — | $e^{\text{cross-entropy}}$; lower is better. | "How many words is the model confused between, on average?" |
| **Byte-Pair Encoding (BPE) / Word-piece** | — | Sub-word tokenization methods. | Splitting "unhappiness" → "un" + "happi" + "ness". |
| **Label smoothing** | $\epsilon_{ls}$ | Softening one-hot targets (e.g., 1.0 → 0.9). | Telling the student "be confident, but not 100% arrogant". |
| **Beam search** | — | Decoding that keeps the top-$k$ partial translations at each step. | Exploring the 4 most promising paths instead of just 1. |
| **FLOPs** | — | Floating-point operations; measure of compute. | The "fuel" consumed by training. |

---

## 3. The Problem: Sequence Transduction Before the Transformer

### 3.1 What Is Sequence Transduction?

Given an input sequence of symbols $(x_1, \dots, x_n)$, produce an output sequence $(y_1, \dots, y_m)$.

**Example (English → German):**

```
Input  (n = 3):  x = ( "I",   "love",  "cats"   )
Output (m = 3):  y = ( "Ich", "liebe", "Katzen" )
```

Most competitive models use an **encoder–decoder** structure:

- The **encoder** maps $(x_1, \dots, x_n)$ to continuous representations $\mathbf{z} = (z_1, \dots, z_n)$.
- The **decoder**, given $\mathbf{z}$, generates $(y_1, \dots, y_m)$ **one symbol at a time**, consuming the previously generated symbols as extra input (auto-regressive).

### 3.2 Recurrent Neural Networks and the Sequential Bottleneck

An RNN computes a hidden state $h_t$ from the previous hidden state and the current input:

$$h_t = f(h_{t-1}, x_t)$$

**Numerical illustration.** For a 50-word sentence:

```
h1 = f(h0, x1)   ─►  h2 = f(h1, x2)  ─►  h3 = f(h2, x3) ─► ... ─► h50
         step 1             step 2              step 3             step 50
```

- You **cannot** compute $h_{50}$ before computing $h_{49}$, which needs $h_{48}$, ... → **50 sequential steps**. GPUs are great at doing thousands of things *at once*, but here they must wait.
- This sequential nature "precludes parallelization within training examples", which becomes critical for long sequences, since memory limits batching across examples.
- Tricks such as factorization and conditional computation improved efficiency, but the **fundamental constraint of sequential computation remained**.

**Second problem — long-range dependencies.** Information from word 1 must travel through 49 transformations to reach word 50. Like a game of "telephone", signals get weaker and gradients vanish.

### 3.3 Attention as an Add-on to RNNs

Attention mechanisms (Bahdanau et al. 2014) allowed the decoder to look directly at all encoder states, modeling dependencies **regardless of distance**. But in almost all cases, attention was **used together with an RNN** — the RNN bottleneck was still there.

> **The Transformer's idea:** keep the attention, remove the RNN.

### 3.4 Convolutional Alternatives: Extended Neural GPU, ByteNet, ConvS2S

To reduce sequential computation, models like the **Extended Neural GPU**, **ByteNet** and **ConvS2S** used convolutions, computing hidden representations for all positions in parallel. But the number of operations needed to relate two distant positions still grows with distance:

| Model | Ops needed to connect positions $i$ and $j$ |
|:---|:---|
| ConvS2S | Grows **linearly** with distance $|i-j|$ |
| ByteNet | Grows **logarithmically** with distance |
| **Transformer** | **Constant** — $O(1)$ |

**Numerical example:** distance between two words = 64 tokens, kernel width $k=3$:

- ConvS2S-style stacking: needs about $64/3 \approx 22$ layers to connect them.
- ByteNet-style dilated convolutions: about $\log_2 64 = 6$ layers.
- Transformer: **1 layer** — every token directly attends to every other token.

The cost of this constant path: attention averages over positions, which can **reduce effective resolution** (blurring). The paper counters this with **Multi-Head Attention** (Section 8).

### 3.5 Self-Attention Before the Transformer

Self-attention (intra-attention) relates different positions of a *single* sequence to compute a representation of that sequence. It had already been used for reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations. **End-to-end memory networks** used recurrent attention instead of sequence-aligned recurrence and did well on simple question answering and language modeling.

> **The Transformer is the first transduction model relying entirely on self-attention**, without sequence-aligned RNNs or convolution.

---

## 4. The Big Picture: Transformer Architecture Overview

```
                                   Output Probabilities
                                           ▲
                                    ┌──────┴──────┐
                                    │   Softmax   │
                                    ├─────────────┤
                                    │   Linear    │
                                    └──────▲──────┘
                                           │
            ENCODER (× N = 6)              │       DECODER (× N = 6)
     ┌──────────────────────────┐   ┌──────┴───────────────────────────┐
     │  ┌────────────────────┐  │   │  ┌─────────────────────────────┐ │
     │  │   Add & Norm       │  │   │  │        Add & Norm           │ │
     │  ├────────────────────┤  │   │  ├─────────────────────────────┤ │
     │  │   Feed Forward     │  │   │  │        Feed Forward         │ │
     │  └─────────▲──────────┘  │   │  └──────────────▲──────────────┘ │
     │  ┌─────────┴──────────┐  │   │  ┌──────────────┴──────────────┐ │
     │  │   Add & Norm       │  │   │  │        Add & Norm           │ │
     │  ├────────────────────┤  │   │  ├─────────────────────────────┤ │
     │  │ Multi-Head         │  │──────►  Multi-Head Attention       │ │
     │  │ Self-Attention     │  │ K,V│  │  (Q from decoder,          │ │
     │  └─────────▲──────────┘  │   │  │   K,V from encoder)         │ │
     └────────────┼─────────────┘   │  └──────────────▲──────────────┘ │
                  │                 │  ┌──────────────┴──────────────┐ │
                  │                 │  │        Add & Norm           │ │
                  │                 │  ├─────────────────────────────┤ │
                  │                 │  │ MASKED Multi-Head           │ │
                  │                 │  │ Self-Attention              │ │
                  │                 │  └──────────────▲──────────────┘ │
                  │                 └─────────────────┼────────────────┘
          (+) Positional Encoding            (+) Positional Encoding
                  ▲                                   ▲
           Input Embedding                    Output Embedding
                  ▲                                   ▲
               Inputs                    Outputs (shifted right)
         "I love cats"                     "<s> Ich liebe"
```

### Base model hyperparameters (memorize these!)

| Symbol | Meaning | Base | Big |
|:---:|:---|:---:|:---:|
| $N$ | Number of layers in encoder (and in decoder) | 6 | 6 |
| $d_{model}$ | Vector size everywhere in the model | 512 | 1024 |
| $d_{ff}$ | Inner size of feed-forward layer | 2048 | 4096 |
| $h$ | Number of attention heads | 8 | 16 |
| $d_k = d_v$ | Size per head $= d_{model}/h$ | 64 | 64 |
| $P_{drop}$ | Dropout rate | 0.1 | 0.3 |
| $\epsilon_{ls}$ | Label smoothing | 0.1 | 0.1 |
| Params | Total parameters | 65 M | 213 M |
| Train steps | — | 100K | 300K |

---

## 5. Step 0: From Words to Vectors (Tokenization & Embeddings)

### 5.1 Tokenization

The paper uses sub-word units:

- **English–German:** byte-pair encoding (BPE) with a **shared source–target vocabulary of ≈37,000 tokens**.
- **English–French:** **32,000 word-piece** vocabulary.

**Example of BPE:** a rare word like `"unbelievably"` might become `["un", "believ", "ably"]`. This means the model never sees a completely unknown word — it can always fall back to smaller pieces.

### 5.2 Embedding lookup

Each token ID is mapped to a learned vector of size $d_{model} = 512$ via an embedding matrix $E \in \mathbb{R}^{V \times d_{model}}$ (with $V \approx 37{,}000$).

**Example:**

```
"I"     → token id 41     → row 41 of E     → [ 0.12, -0.53, 0.07, ..., 0.33]  (512 numbers)
"love"  → token id 1187   → row 1187 of E   → [-0.41,  0.22, 0.91, ..., -0.08] (512 numbers)
"cats"  → token id 9032   → row 9032 of E   → [ 0.05,  0.66, -0.19, ..., 0.27] (512 numbers)
```

So the sentence becomes a matrix of shape $3 \times 512$.

### 5.3 Scaling the embeddings by $\sqrt{d_{model}}$

In the embedding layers, the weights are **multiplied by $\sqrt{d_{model}}$**:

$$\sqrt{512} \approx 22.63$$

**Why?** Embedding values are typically small (e.g., around $\pm 0.04$), while positional encodings are in $[-1, 1]$. Multiplying by 22.63 brings the embedding magnitude up so that the word meaning isn't drowned out by the positional signal when they're added.

Example: an embedding entry of $0.04$ becomes $0.04 \times 22.63 \approx 0.905$, comparable in size to a PE value like $0.84$.

### 5.4 Shared weights

The paper **shares the same weight matrix** between:
1. the encoder input embedding,
2. the decoder output embedding,
3. the pre-softmax linear transformation (output layer).

(This is possible for En–De because of the shared BPE vocabulary.) This saves parameters: $37{,}000 \times 512 \approx 18.9$M parameters are stored only once instead of three times. See Section 14.

---

## 6. Positional Encoding: Giving the Model a Sense of Order

### 6.1 Why do we need it?

Self-attention treats its input as a **set**, not a sequence. Without position information:

```
"dog bites man"   and   "man bites dog"
```

would produce the **same** set of attention interactions (just permuted). The meanings are obviously different! So we must inject order information.

### 6.2 The formula

The positional encoding has the same dimension $d_{model}$ as the embeddings so they can be **summed**:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- $pos$ = position in the sentence (0, 1, 2, ...).
- $i$ = dimension-pair index ($0 \le i < d_{model}/2$). Even dimensions use sine, odd dimensions use cosine.
- Each dimension pair corresponds to a sinusoid with its own frequency. The **wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$**.

### 6.3 Numerical example with a tiny $d_{model} = 4$

There are 2 frequency pairs: $i=0$ with divisor $10000^{0/4} = 1$, and $i=1$ with divisor $10000^{2/4} = 100$.

| pos | dim 0: $\sin(pos/1)$ | dim 1: $\cos(pos/1)$ | dim 2: $\sin(pos/100)$ | dim 3: $\cos(pos/100)$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

Observations:
- Dims 0–1 change **fast** (high frequency) → distinguish neighboring positions.
- Dims 2–3 change **slowly** (low frequency) → encode coarse position over long ranges.
- It's like a clock: the seconds hand (fast) and the hour hand (slow) together tell exact time.

### 6.4 The real model ($d_{model} = 512$)

For $pos = 1$:

```
dim 0  = sin(1)                     = 0.8415
dim 1  = cos(1)                     = 0.5403
dim 2  = sin(1 / 10000^(2/512))     = 0.8219
dim 3  = cos(1 / 10000^(2/512))     = 0.5697
dim 4  = sin(1 / 10000^(4/512))     = 0.8020
dim 5  = cos(...)                   = 0.5974
 ...
dim 510 = sin(1 / 10000^(510/512))  ≈ 0.0001
dim 511 = cos(...)                  ≈ 1.0000
```

Wavelengths: the first pair has wavelength $2\pi \approx 6.28$ positions; the last pair has wavelength $2\pi \cdot 10000^{510/512} \approx 60{,}611$ positions.

### 6.5 Adding to the embedding

$$\text{input}_{pos} = \sqrt{d_{model}} \cdot \text{Embedding}(token_{pos}) + PE_{pos}$$

Example ($d_{model}=4$, token "love" at position 1, scaled embedding $[0.50, -0.20, 0.30, 0.10]$):

$$[0.50, -0.20, 0.30, 0.10] + [0.8415, 0.5403, 0.0100, 1.0000] = [1.3415, 0.3403, 0.3100, 1.1000]$$

### 6.6 Why sinusoids? The relative-position property

The authors hypothesized this lets the model easily learn to attend by **relative positions**, because for any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$. For one frequency $\omega$:

$$\begin{bmatrix}\sin(\omega(p+k)) \\ \cos(\omega(p+k))\end{bmatrix} = \begin{bmatrix}\cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k)\end{bmatrix} \begin{bmatrix}\sin(\omega p) \\ \cos(\omega p)\end{bmatrix}$$

The matrix depends only on $k$, not on $p$ — a **rotation**.

**Numerical check** ($\omega = 1$, $p = 1$, $k = 1$):
$\sin(2) = \sin(1)\cos(1) + \cos(1)\sin(1) = 0.8415 \times 0.5403 + 0.5403 \times 0.8415 = 0.9093$ ✓ (matches the table above).

### 6.7 Learned vs. sinusoidal

The authors also tried **learned positional embeddings** and found **nearly identical results** (Table 3, row E: BLEU 25.7 vs. 25.8). They chose sinusoids because they **may allow the model to extrapolate to sequence lengths longer than those seen during training** (a learned table has no entry for position 600 if trained only up to 512).

---

## 7. Attention: The Core Idea

> "An attention function can be described as mapping a **query** and a set of **key-value pairs** to an output, where the query, keys, values, and output are all vectors. The output is computed as a **weighted sum of the values**, where the weight assigned to each value is computed by a **compatibility function** of the query with the corresponding key."

### 7.1 Queries, Keys and Values (The Library Analogy)

```
You walk into a library with a QUESTION (Query): "books about cats"

Each book has a LABEL (Key) and CONTENT (Value):
   Book A: Key = "dogs"        Value = <dog content>
   Book B: Key = "cats & pets" Value = <cat content>
   Book C: Key = "cooking"     Value = <recipes>

1. Compare query with each key    → similarity: A = 2, B = 9, C = 0
2. Softmax similarities           → weights:    A ≈ 0.001, B ≈ 0.999, C ≈ 0.000
3. Weighted sum of values         → output ≈ mostly the cat content
```

In the Transformer, each token creates **its own** query, key and value by multiplying its vector by three learned matrices $W^Q, W^K, W^V$.

### 7.2 Scaled Dot-Product Attention

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V}$$

```
   Q ──┐
       ├─► MatMul ─► Scale (÷√dk) ─► Mask (optional, decoder) ─► SoftMax ─┐
   K ──┘                                                                  ├─► MatMul ─► Output
   V ─────────────────────────────────────────────────────────────────────┘
```

Shapes (for $n$ tokens):
- $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$
- $QK^T \in \mathbb{R}^{n \times n}$: score of every token against every other token.
- softmax applied **row-wise** → each row sums to 1.
- Output $\in \mathbb{R}^{n \times d_v}$.

In practice, attention is computed on **a set of queries simultaneously**, packed into matrices — that's why it's so fast: it's just two matrix multiplications.

### 7.3 Full Numerical Walkthrough of Scaled Dot-Product Attention

Toy setup: 3 tokens, $d_{model} = 4$, $d_k = d_v = 2$.

**Input matrix $X$** (embeddings + positional encodings, one row per token):

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix} \quad \begin{matrix} \leftarrow \text{"I"} \\ \leftarrow \text{"love"} \\ \leftarrow \text{"cats"} \end{matrix}$$

**Learned projection matrices** ($4 \times 2$ each):

$$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix}$$

**Step 1 — Compute Q, K, V:**

For "I" (row $[1,0,1,0]$): $q = [1\cdot1 + 1\cdot1,\ 1\cdot0 + 1\cdot1] = [2, 1]$.

$$Q = XW^Q = \begin{bmatrix} 2 & 1 \\ 0 & 2 \\ 2 & 2 \end{bmatrix}, \quad K = XW^K = \begin{bmatrix} 1 & 1 \\ 2 & 1 \\ 2 & 2 \end{bmatrix}, \quad V = XW^V = \begin{bmatrix} 1 & 0 \\ 1 & 3 \\ 2 & 2 \end{bmatrix}$$

**Step 2 — Raw scores $QK^T$:**

Entry (I, love) $= q_I \cdot k_{love} = 2\cdot2 + 1\cdot1 = 5$.

$$QK^T = \begin{bmatrix} 3 & 5 & 6 \\ 2 & 2 & 4 \\ 4 & 6 & 8 \end{bmatrix}$$

**Step 3 — Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.4142$:**

$$\frac{QK^T}{\sqrt{2}} = \begin{bmatrix} 2.1213 & 3.5355 & 4.2426 \\ 1.4142 & 1.4142 & 2.8284 \\ 2.8284 & 4.2426 & 5.6569 \end{bmatrix}$$

**Step 4 — Row-wise softmax:**

Row "I": $e^{2.1213} = 8.34$, $e^{3.5355} = 34.31$, $e^{4.2426} = 69.59$. Sum $= 112.24$.
Weights $= [8.34/112.24,\ 34.31/112.24,\ 69.59/112.24] = [0.0743,\ 0.3057,\ 0.6200]$.

$$A = \begin{bmatrix} 0.0743 & 0.3057 & 0.6200 \\ 0.1636 & 0.1636 & 0.6728 \\ 0.0454 & 0.1867 & 0.7679 \end{bmatrix}$$

Interpretation: "I" pays 62% attention to "cats", 31% to "love", 7% to itself.

**Step 5 — Weighted sum of values $AV$:**

Output for "I" $= 0.0743\cdot[1,0] + 0.3057\cdot[1,3] + 0.6200\cdot[2,2]$
$= [0.0743 + 0.3057 + 1.2400,\ 0 + 0.9171 + 1.2400] = [1.6200,\ 2.1571]$.

$$\text{Output} = AV = \begin{bmatrix} 1.6200 & 2.1571 \\ 1.6728 & 1.8364 \\ 1.7679 & 2.0959 \end{bmatrix}$$

Each token's new vector is now a **context-aware mixture** of the whole sentence. 🎉

### 7.4 Why Divide by $\sqrt{d_k}$? (The Scaling Factor)

**The paper's argument (footnote 4):** assume the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Then

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i \quad \text{has mean } 0 \text{ and variance } d_k.$$

With $d_k = 64$, the standard deviation of dot products is $\sqrt{64} = 8$. (A simulation with 100,000 random pairs gives variance ≈ 63.98 ✓; after dividing by 8, variance ≈ 1.00 ✓.)

**Why large values are bad — numerical example:**

| Scores | Softmax result | Comment |
|:---|:---|:---|
| $[8, 0, -8]$ (unscaled, typical std 8) | $[0.9997, 0.0003, 0.0000]$ | Almost one-hot → **gradients ≈ 0** (saturated) |
| $[1, 0, -1]$ (scaled by $1/8$) | $[0.6652, 0.2447, 0.0900]$ | Smooth → healthy gradients |

When softmax is pushed into regions where one value dominates, its gradient becomes extremely small, so learning stalls. Dividing by $\sqrt{d_k}$ keeps the variance at ≈1 regardless of $d_k$.

### 7.5 Dot-Product vs. Additive Attention

The two most common attention functions:

| Type | Score formula | Notes |
|:---|:---|:---|
| **Additive** (Bahdanau) | $v^T \tanh(W_1 q + W_2 k)$ | Uses a feed-forward network with one hidden layer. |
| **Dot-product** (multiplicative) | $q \cdot k$ | Identical to the paper's, minus the scaling. |

- Theoretical complexity is similar, but **dot-product is much faster and more space-efficient** because it uses highly optimized matrix multiplication code.
- For small $d_k$ both perform similarly. For **large $d_k$**, additive attention outperforms unscaled dot-product — the authors suspect it's because of the large-magnitude problem above. Hence the **scaling by $1/\sqrt{d_k}$** fixes dot-product attention.

---

## 8. Multi-Head Attention

### 8.1 Why Multiple Heads?

With a single head, attention averages everything into one weighted mix, which "inhibits" the model from attending to different things at once. Multi-head attention lets the model **jointly attend to information from different representation subspaces at different positions**.

**Example sentence:** *"The animal didn't cross the street because **it** was too tired."*

```
Head 1: "it" → "animal"     (coreference: who is "it"?)
Head 2: "it" → "tired"      (what property is described?)
Head 3: "it" → "was"        (syntax: subject–verb)
...
Head 8: "it" → "because"    (discourse structure)
```

One averaged head would mix these into a blur; 8 heads keep them separate.

### 8.2 The Math

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

Projection matrices:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ → $512 \times 64$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ → $512 \times 64$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ → $512 \times 64$
- $W^O \in \mathbb{R}^{h d_v \times d_{model}}$ → $512 \times 512$

With $h = 8$: $d_k = d_v = d_{model}/h = 512/8 = 64$.

```
          ┌──► Linear(Wq1,Wk1,Wv1) ─► Scaled Dot-Product Attention ─► head1 (n×64) ─┐
          ├──► Linear(Wq2,Wk2,Wv2) ─► Scaled Dot-Product Attention ─► head2 (n×64) ─┤
 Q,K,V ───┤          ...                       ...                                  ├─► Concat (n×512) ─► Linear W^O ─► (n×512)
          └──► Linear(Wq8,Wk8,Wv8) ─► Scaled Dot-Product Attention ─► head8 (n×64) ─┘
```

### 8.3 Numerical Walkthrough (2 Heads)

Same toy $X$ ($3 \times 4$), now with $h = 2$, $d_k = d_v = 4/2 = 2$.

**Head 1** = exactly the computation from Section 7.3:

$$\text{head}_1 = \begin{bmatrix} 1.6200 & 2.1571 \\ 1.6728 & 1.8364 \\ 1.7679 & 2.0959 \end{bmatrix}$$

**Head 2** uses *different* matrices:

$$W_2^Q = \begin{bmatrix} 0&1\\1&0\\0&0\\1&1 \end{bmatrix},\ W_2^K = \begin{bmatrix} 1&0\\0&0\\0&1\\1&1 \end{bmatrix},\ W_2^V = \begin{bmatrix} 0&1\\1&1\\1&0\\0&0 \end{bmatrix}$$

$$Q_2 = \begin{bmatrix} 0&1\\3&1\\2&2 \end{bmatrix},\ K_2 = \begin{bmatrix} 1&1\\1&1\\2&2 \end{bmatrix},\ V_2 = \begin{bmatrix} 1&1\\2&2\\2&2 \end{bmatrix}$$

Scores $Q_2K_2^T = \begin{bmatrix}1&1&2\\4&4&8\\4&4&8\end{bmatrix}$, scaled by $1/\sqrt2$ → softmax:

$$A_2 = \begin{bmatrix} 0.2483 & 0.2483 & 0.5035 \\ 0.0529 & 0.0529 & 0.8943 \\ 0.0529 & 0.0529 & 0.8943 \end{bmatrix} \Rightarrow \text{head}_2 = A_2V_2 = \begin{bmatrix} 1.7517 & 1.7517 \\ 1.9471 & 1.9471 \\ 1.9471 & 1.9471 \end{bmatrix}$$

Notice Head 2 learned a **different attention pattern** (e.g., "I" spreads attention more evenly: 25%/25%/50%).

**Concatenate** ($3 \times 4$):

$$\text{Concat} = \begin{bmatrix} 1.6200 & 2.1571 & 1.7517 & 1.7517 \\ 1.6728 & 1.8364 & 1.9471 & 1.9471 \\ 1.7679 & 2.0959 & 1.9471 & 1.9471 \end{bmatrix}$$

**Output projection** with (for simplicity) $W^O = 0.5 \cdot I_4$:

$$\text{MultiHead} = \begin{bmatrix} 0.8100 & 1.0785 & 0.8759 & 0.8759 \\ 0.8364 & 0.9182 & 0.9736 & 0.9736 \\ 0.8840 & 1.0479 & 0.9736 & 0.9736 \end{bmatrix}$$

The output has shape $3 \times 4$ — **same as the input** — which is essential for the residual connection (Section 11).

### 8.4 Tensor Shapes in the Real Model

For a sentence of $n = 10$ tokens, base model:

| Step | Shape |
|:---|:---|
| Input $X$ | $10 \times 512$ |
| Per head: $Q_i, K_i, V_i$ | $10 \times 64$ each |
| Per head: scores $Q_iK_i^T$ | $10 \times 10$ |
| Per head: output | $10 \times 64$ |
| Concat 8 heads | $10 \times 512$ |
| After $W^O$ ($512 \times 512$) | $10 \times 512$ |

With a batch of $B$ sentences, everything gets an extra leading dimension $B$, and heads are usually computed as one tensor of shape $B \times 8 \times 10 \times 64$.

### 8.5 Computational Cost of Multi-Head vs. Single-Head

Because each head has reduced dimension, **total cost is similar to single-head attention with full dimensionality**:

- Single head, $d_k = 512$: score cost $\propto n^2 \times 512$.
- 8 heads, $d_k = 64$: score cost $\propto 8 \times n^2 \times 64 = n^2 \times 512$. **Same!**

You get 8 different "perspectives" for free.

---

## 9. The Three Ways Attention Is Used in the Transformer

| Where | Queries from | Keys & Values from | Mask? |
|:---|:---|:---|:---:|
| Encoder self-attention | Previous encoder layer | Previous encoder layer | No (only padding) |
| Decoder masked self-attention | Previous decoder layer | Previous decoder layer (positions ≤ current) | **Yes** |
| Encoder–decoder attention | Previous decoder layer | **Final encoder output** | No |

### 9.1 Encoder Self-Attention

All keys, values and queries come from the output of the previous encoder layer. **Each position can attend to all positions** in the previous layer. "cats" can look at "I" and "love", and vice-versa.

### 9.2 Masked Decoder Self-Attention (with Numbers)

In the decoder, each position may attend only to positions **up to and including itself**. We must prevent **leftward information flow** to keep the auto-regressive property — during training, the model must not "see the answer" (future words).

Implementation: inside scaled dot-product attention, **set to $-\infty$ all values in the softmax input that correspond to illegal connections**. Since $e^{-\infty} = 0$, they get weight 0.

**Numerical example** — reuse the scaled scores from Section 7.3 and apply a look-ahead mask:

$$\text{masked scores} = \begin{bmatrix} 2.1213 & -\infty & -\infty \\ 1.4142 & 1.4142 & -\infty \\ 2.8284 & 4.2426 & 5.6569 \end{bmatrix}$$

Softmax row-wise:

$$A_{masked} = \begin{bmatrix} 1.0000 & 0 & 0 \\ 0.5000 & 0.5000 & 0 \\ 0.0454 & 0.1867 & 0.7679 \end{bmatrix}$$

- Token 1 can only see itself → weight 1.0.
- Token 2 sees tokens 1 and 2 → equal scores → 50%/50%.
- Token 3 sees everything (unchanged from before).

Outputs: $A_{masked}V = \begin{bmatrix} 1.0000 & 0.0000 \\ 1.0000 & 1.5000 \\ 1.7679 & 2.0959 \end{bmatrix}$.

**Mask matrix pattern for 5 positions** (1 = allowed, 0 = blocked → $-\infty$):

```
           k1  k2  k3  k4  k5
     q1  [  1   0   0   0   0 ]
     q2  [  1   1   0   0   0 ]
     q3  [  1   1   1   0   0 ]
     q4  [  1   1   1   1   0 ]
     q5  [  1   1   1   1   1 ]
```

**"Outputs (shifted right)":** The output embeddings are offset by one position. Combined with the mask, this ensures **the prediction for position $i$ depends only on known outputs at positions less than $i$**.

```
Decoder input  (shifted right):  <s>    Ich    liebe
Decoder target (to predict):     Ich    liebe  Katzen
Position i uses:                 {<s>}  {<s>,Ich}  {<s>,Ich,liebe}
```

This lets the decoder train on **all positions in parallel** (teacher forcing), while still behaving auto-regressively.

### 9.3 Encoder–Decoder (Cross) Attention

- **Queries** come from the previous decoder layer ("what am I looking for in the source?").
- **Keys and values** come from the **output of the encoder** (the "memory").
- Every decoder position can attend over **all positions in the input sequence**. This mimics classic seq2seq attention.

**Shape example:** source "I love cats" ($n = 3$), target so far `"<s> Ich liebe Katzen"` ($m = 4$):

- $Q$: $4 \times 64$ (from decoder), $K, V$: $3 \times 64$ (from encoder).
- Scores: $4 \times 3$ — each German token gets a distribution over the 3 English tokens.

Example (plausible learned weights):

```
              "I"    "love"   "cats"
"<s>"       [ 0.80    0.15     0.05 ]   → about to produce "Ich"
"Ich"       [ 0.10    0.85     0.05 ]   → about to produce "liebe"
"liebe"     [ 0.05    0.10     0.85 ]   → about to produce "Katzen"
"Katzen"    [ 0.30    0.30     0.40 ]   → about to produce "</s>"
```

---

## 10. Position-wise Feed-Forward Networks

Each encoder and decoder layer contains a fully connected FFN, applied **to each position separately and identically**:

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)\, W_2 + b_2$$

- Two linear transformations with a **ReLU** in between.
- $W_1 \in \mathbb{R}^{512 \times 2048}$, $W_2 \in \mathbb{R}^{2048 \times 512}$. Inner dimension $d_{ff} = 2048$ (4× expansion).
- **Same parameters across positions** within a layer, but **different parameters from layer to layer**.
- Equivalent description: **two convolutions with kernel size 1**.

**Intuition:** attention = words *communicate*; FFN = each word *processes* the gathered information on its own.

**Numerical example** (toy $d_{model}=2$, $d_{ff}=4$):

$x = [1, 2]$, 
$W_1 = \begin{bmatrix} 1 & -1 & 0 & 2 \\ 0 & 1 & -1 & 1 \end{bmatrix}$, $b_1 = [0, 0, 0, -1]$,
$W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0.5 & -0.5 \end{bmatrix}$, $b_2 = [0, 0]$.

1. $xW_1 + b_1 = [1,\ -1+2,\ -2,\ 2+2] + [0,0,0,-1] = [1,\ 1,\ -2,\ 3]$
2. ReLU: $[1,\ 1,\ 0,\ 3]$ (the negative $-2$ is zeroed)
3. $\times W_2$: $[1 + 0 + 0 + 1.5,\ 0 + 1 + 0 - 1.5] = [2.5,\ -0.5]$

Output has the same size as the input (2), ready for the residual connection.

**Parameter count per FFN (real model):** $512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712$ ≈ **2.1M** — twice as many as the attention sub-layer (1.05M).

---

## 11. Residual Connections & Layer Normalization

Every sub-layer (attention or FFN) is wrapped like this:

$$\boxed{\text{output} = \text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))}$$

### 11.1 Residual connection

Adds the input back to the output: $x + \text{Sublayer}(x)$. Benefits:
- Gradients flow directly through the "+" shortcut → deep stacks train easily.
- The sub-layer only needs to learn a *correction* to $x$.

**To make the addition possible, all sub-layers and embedding layers produce outputs of dimension $d_{model} = 512$.**

### 11.2 Layer normalization

For a vector $x$ of size $d$:

$$\mu = \frac{1}{d}\sum_i x_i,\quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2,\quad \text{LN}(x)_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i$$

($\gamma, \beta$ are learned, initialized to 1 and 0.)

**Simple numerical example:** $x = [2, 4, 6, 8]$
- $\mu = 5$, $\sigma^2 = \frac{9 + 1 + 1 + 9}{4} = 5$, $\sigma = 2.236$
- $\text{LN}(x) = [-3, -1, 1, 3]/2.236 = [-1.342,\ -0.447,\ 0.447,\ 1.342]$

**Continuing our toy example** (Sections 7–8): residual for "I":
$x + \text{MultiHead} = [1,0,1,0] + [0.81, 1.0785, 0.8759, 0.8759] = [1.81, 1.0785, 1.8759, 0.8759]$
- $\mu = 1.4101$, $\sigma^2 = 0.193$
- $\text{LN} = [0.910,\ -0.755,\ 1.060,\ -1.216]$

> **Note:** The paper uses **Post-LN** (normalize *after* adding). Many later models (GPT-2, etc.) moved to **Pre-LN** ($x + \text{Sublayer}(\text{LN}(x))$) for more stable training of very deep models — but the original is Post-LN.

---

## 12. The Encoder Stack in Detail

- A stack of $N = 6$ **identical layers** (identical structure, separate weights).
- Each layer has **two sub-layers**:
  1. Multi-head self-attention
  2. Position-wise fully connected feed-forward network
- Each sub-layer: residual connection + layer normalization.

```
x ──► [Multi-Head Self-Attn] ──► (+x) ──► LayerNorm ──► y
y ──► [Feed-Forward]         ──► (+y) ──► LayerNorm ──► output of layer
```

**Shape journey for "I love cats" (n = 3):**

```
Tokens                              (3,)
Embedding × √512 + PE               (3, 512)
Encoder layer 1 → ... → layer 6     (3, 512) at every layer
Final encoder output  z (memory)    (3, 512)   → used by every decoder layer
```

---

## 13. The Decoder Stack in Detail

- Also $N = 6$ identical layers.
- Each layer has **three sub-layers**:
  1. **Masked** multi-head self-attention (over previously generated outputs)
  2. Multi-head attention over the **encoder output** (cross-attention)
  3. Position-wise feed-forward network
- Each sub-layer: residual + layer normalization.

```
y ──► [Masked Self-Attn]                    ──► (+y) ──► LN ──► a
a ──► [Cross-Attn: Q=a, K=V=encoder output]  ──► (+a) ──► LN ──► b
b ──► [Feed-Forward]                         ──► (+b) ──► LN ──► output of layer
```

---

## 14. The Output Layer: Linear + Softmax & Weight Sharing

The decoder output (size 512 per position) is converted to next-token probabilities using:

1. A **learned linear transformation** to vocabulary size: $512 \to 37{,}000$ scores (logits).
2. **Softmax** → probability distribution over all tokens.

**Example (tiny vocabulary of 5):**

```
Decoder vector at last position  ──► Linear ──► logits = [1.2, 0.3, 4.1, -0.5, 0.9]
                                                 vocab = [der, die, Katzen, Hund, und]
softmax → [0.049, 0.020, 0.885, 0.009, 0.036]   → predict "Katzen" (88.5%)
```

**Weight sharing:** the pre-softmax linear weights are the **same matrix** as the embedding matrix (transposed). Intuition: a token's output score = dot product between the decoder vector and that token's embedding — "how similar is my hidden state to the meaning of each word?". In the embedding layers, these weights are multiplied by $\sqrt{d_{model}}$.

---

## 15. Counting Parameters: Where Do 65M and 213M Come From?

Let's verify the base model size (with biases, $V = 37{,}000$, shared embeddings):

| Component | Formula | Count |
|:---|:---|---:|
| Multi-head attention | $4 \times (512 \times 512 + 512)$ ($W^Q, W^K, W^V, W^O$) | 1,050,624 |
| FFN | $512 \times 2048 + 2048 + 2048 \times 512 + 512$ | 2,099,712 |
| LayerNorm | $2 \times 512$ ($\gamma, \beta$) | 1,024 |
| **Encoder layer** | 1 attn + 1 FFN + 2 LN | **3,152,384** |
| **Decoder layer** | 2 attn + 1 FFN + 3 LN | **4,204,032** |
| 6 encoder layers | | 18,914,304 |
| 6 decoder layers | | 25,224,192 |
| Shared embedding | $37{,}000 \times 512$ | 18,944,000 |
| **Total** | | **≈ 63.1M** |

≈ 63M — very close to the paper's **65M** (the exact number depends on vocabulary size and implementation details). Doing the same for the big model ($d_{model}=1024$, $d_{ff}=4096$) gives ≈ 214M vs. paper's **213M** ✓.

Note the per-head projections ($W_i^Q$ for 8 heads, each $512 \times 64$) together equal one $512 \times 512$ matrix — that's why the attention count is $4 \times 512^2$.

---

## 16. Why Self-Attention? (Complexity, Parallelism, Path Length)

The paper compares self-attention layers to recurrent and convolutional layers mapping a sequence $(x_1..x_n)$ to $(z_1..z_n)$ with $x_i, z_i \in \mathbb{R}^d$, using **three desiderata**:

1. **Total computational complexity per layer.**
2. **Amount of computation that can be parallelized** (minimum number of sequential operations).
3. **Path length between long-range dependencies** — the shorter the paths forward and backward signals must traverse, the easier it is to learn long-range dependencies.

### Table 1 (from the paper)

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:---|:---:|:---:|:---:|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

($n$ = sequence length, $d$ = representation dimension, $k$ = kernel size, $r$ = neighborhood size.)

### Numerical comparison of complexity ($d = 512$)

| Sentence length $n$ | Self-attention $n^2 d$ | Recurrent $n d^2$ | Winner |
|:---:|---:|---:|:---|
| 30 | 460,800 | 7,864,320 | Self-attention (17× cheaper) |
| 512 | 134,217,728 | 134,217,728 | Tie ($n = d$) |
| 2,000 | 2,048,000,000 | 524,288,000 | Recurrent (4× cheaper) |

So **self-attention is faster than recurrent layers when $n < d$**, which is most often the case for sentence representations used by state-of-the-art translation models (word-piece / byte-pair). Typical sentences have 20–50 tokens ≪ 512.

### Sequential operations & path length ($n = 100$)

| Layer | Sequential ops | Max path between token 1 and 100 |
|:---|:---:|:---|
| RNN | 100 | 100 steps |
| Conv, contiguous kernels $k=3$ | 1 per layer | needs $O(n/k) \approx 34$ stacked layers |
| Conv, dilated $k=3$ | 1 per layer | $\log_3 100 \approx 4.2$ layers |
| **Self-attention** | **1** | **1** |

### Notes on convolutions
- A single conv layer with $k < n$ does **not** connect all pairs of positions.
- Convolutional layers are generally **more expensive than recurrent layers by a factor of $k$**.
- **Separable convolutions** reduce complexity to $O(k \cdot n \cdot d + n \cdot d^2)$. Even with $k = n$, that equals the **combination of a self-attention layer and a point-wise FFN** — exactly the Transformer's approach.

### Restricted self-attention
For very long sequences, self-attention can be restricted to a neighborhood of size $r$ around each position: complexity drops to $O(r n d)$ but path length grows to $O(n/r)$. Example: $n = 10{,}000$, $r = 100$ → each layer only computes $100$ scores per token instead of $10{,}000$, but connecting the extremes takes ≈100 layers. The authors left this for future work.

### Side benefit: interpretability
Self-attention can yield **more interpretable models**: individual heads clearly learn different tasks, and many show behavior related to syntactic and semantic structure (Section 22).

---

## 17. Training Setup

### 17.1 Data and Batching

| Dataset | Size | Vocabulary |
|:---|:---|:---|
| WMT 2014 English–German | ≈4.5 million sentence pairs | BPE, shared source–target ≈37,000 tokens |
| WMT 2014 English–French | 36 million sentences | 32,000 word-piece vocabulary |

- Sentence pairs are **batched together by approximate sequence length** (to minimize padding).
- Each training batch contains roughly **25,000 source tokens and 25,000 target tokens**.

Example: if sentences average 25 tokens, one batch ≈ $25{,}000 / 25 = 1{,}000$ sentence pairs.

### 17.2 Hardware and Schedule

- One machine with **8 NVIDIA P100 GPUs**.
- **Base model:** each step ≈ **0.4 s**; trained for **100,000 steps** → $100{,}000 \times 0.4 = 40{,}000$ s ≈ **12 hours**.
- **Big model:** each step ≈ **1.0 s**; trained for **300,000 steps** → $300{,}000$ s ≈ **3.5 days**.

### 17.3 The Adam Optimizer & the Warmup Learning-Rate Schedule

**Adam** with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$.

**Learning rate schedule:**

$$lrate = d_{model}^{-0.5} \cdot \min\left(step\_num^{-0.5},\ step\_num \cdot warmup\_steps^{-1.5}\right), \quad warmup\_steps = 4000$$

- For the first 4000 steps: LR increases **linearly** (the second term is smaller).
- After that: LR decreases **proportionally to the inverse square root** of the step number.

**Numerical table** ($d_{model} = 512$, so $512^{-0.5} = 0.04419$):

| step | lrate | Phase |
|---:|---:|:---|
| 1 | 0.000000175 | warmup (tiny) |
| 100 | 0.0000175 | warmup |
| 1,000 | 0.000175 | warmup |
| **4,000** | **0.000699** (peak) | switch point |
| 8,000 | 0.000494 | decay |
| 20,000 | 0.000313 | decay |
| 100,000 | 0.000140 | decay (end of base training) |

```
lrate
 7e-4 |          *
      |        *   *
 5e-4 |      *       *  *
      |    *               *   *
 3e-4 |  *                          *    *
      | *                                     *      *
 1e-4 |*                                                   *
      +----------------------------------------------------------► step
      0     4000                                          100k
```

**Why warmup?** At the start, weights are random and Adam's statistics are unreliable; a big learning rate could make training diverge. Starting small and ramping up stabilizes early training.

### 17.4 Regularization: Residual Dropout

Dropout with $P_{drop} = 0.1$ (base) is applied:
1. To the **output of each sub-layer, before** it is added to the sub-layer input and normalized.
2. To the **sums of the embeddings and positional encodings** in both encoder and decoder stacks.

Example: with $P_{drop} = 0.1$, a sub-layer output $[0.8, 1.1, 0.9, 0.9]$ might become $[0.889, 0, 1.0, 1.0]$ during training (one entry zeroed, the rest scaled by $1/0.9$ to keep the expected value the same).

### 17.5 Regularization: Label Smoothing

Label smoothing with $\epsilon_{ls} = 0.1$. Instead of training toward a one-hot target (100% on the correct word), the target puts ≈90% on the correct word and spreads ≈10% over the others.

**Numerical example (vocabulary of 5, correct word = index 2):**

- Hard target: $[0, 0, 1, 0, 0]$
- Smoothed target (spread $0.1$ uniformly over 5 tokens): $[0.02, 0.02, 0.92, 0.02, 0.02]$

Model prediction $p = [0.05, 0.05, 0.80, 0.05, 0.05]$:
- Hard cross-entropy: $-\ln(0.80) = 0.2231$
- Smoothed cross-entropy: $-(0.92\ln0.8 + 4 \times 0.02 \ln 0.05) = 0.2053 + 0.2397 = 0.4450$

The smoothed loss **penalizes the model for being too confident** — pushing $p_{correct} \to 1.0$ would make the $0.02 \ln p_{other}$ terms explode.

> **Paper's finding:** label smoothing **hurts perplexity** (the model learns to be more unsure), but **improves accuracy and BLEU**. Table 3: $\epsilon_{ls}=0$ gives PPL 4.67 (better) but BLEU 25.3 (worse) vs. 25.8 with $\epsilon_{ls}=0.1$.

---

## 18. Inference: Autoregressive Decoding, Beam Search, Checkpoint Averaging

### 18.1 Checkpoint averaging
- **Base models:** a single model obtained by **averaging the last 5 checkpoints**, written at 10-minute intervals.
- **Big models:** averaging the **last 20 checkpoints**.

Example: if a weight has values $0.51, 0.49, 0.50, 0.52, 0.48$ in the last 5 checkpoints, the final model uses $0.50$. Averaging smooths out noise from the last training steps — like a cheap ensemble.

### 18.2 Beam search
- **Beam size = 4**, **length penalty $\alpha = 0.6$** (hyperparameters chosen on the development set).
- Maximum output length = **input length + 50**, but terminate early when possible.

**Why a length penalty?** Log-probabilities are negative and add up, so longer sentences always get lower scores; without correction, beam search prefers too-short translations. Using the GNMT-style penalty $lp(Y) = \frac{(5 + |Y|)^\alpha}{(5+1)^\alpha}$, score $= \log P(Y)/lp(Y)$:

| Hypothesis | Length | $\log P$ | $lp$ | Score |
|:---|:---:|:---:|:---:|:---:|
| A (short) | 5 | −4.0 | $(10/6)^{0.6} = 1.359$ | −2.944 |
| B (long) | 10 | −5.0 | $(15/6)^{0.6} = 1.733$ | **−2.885** ✅ |

Without the penalty A wins (−4.0 > −5.0); with it, the more complete B wins.

### 18.3 Beam search example (beam = 2 for brevity)

```
Step 1:  <s> → "Ich" (−0.1), "Mir" (−2.5)                         keep both
Step 2:  "Ich liebe" (−0.3), "Ich mag" (−1.4), "Mir gefallen" (−3.1) ... keep top 2
Step 3:  "Ich liebe Katzen" (−0.5), "Ich mag Katzen" (−1.7)      keep top 2
Step 4:  "Ich liebe Katzen </s>" (−0.6)  → finished, best hypothesis
```

---

## 19. Results: Machine Translation

### Table 2 (from the paper): BLEU on newstest2014 and training cost

| Model | EN-DE BLEU | EN-FR BLEU | Training Cost EN-DE (FLOPs) | Training Cost EN-FR (FLOPs) |
|:---|:---:|:---:|:---:|:---:|
| ByteNet | 23.75 | — | — | — |
| Deep-Att + PosUnk | — | 39.2 | — | $1.0 \times 10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3 \times 10^{19}$ | $1.4 \times 10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6 \times 10^{18}$ | $1.5 \times 10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0 \times 10^{19}$ | $1.2 \times 10^{20}$ |
| Deep-Att + PosUnk Ensemble | — | 40.4 | — | $8.0 \times 10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8 \times 10^{20}$ | $1.1 \times 10^{21}$ |
| ConvS2S Ensemble | 26.36 | **41.29** | $7.7 \times 10^{19}$ | $1.2 \times 10^{21}$ |
| **Transformer (base)** | 27.3 | 38.1 | $\mathbf{3.3 \times 10^{18}}$ | $3.3 \times 10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ | $2.3 \times 10^{19}$ |

### Key takeaways

- **EN→DE:** the big Transformer gets **28.4 BLEU**, beating the best previous models **including ensembles by more than 2.0 BLEU** (28.4 vs. 26.36). Even the **base model** (27.3) beats all previous models and ensembles at a **fraction of the training cost**.
- **EN→FR:** the big model reaches **41.8 BLEU** (single-model SOTA) at **less than 1/4 the training cost** of the previous state-of-the-art model. For EN→FR, the big model used $P_{drop} = 0.1$ instead of 0.3.

> **Note on v7 of the paper:** the abstract and Table 2 report **41.8** on EN-FR, while the text of Section 6.1 still says **41.0** — a leftover from an earlier version. 41.8 is the updated figure.

### How training cost (FLOPs) is estimated

$$\text{FLOPs} = \text{training time} \times \text{number of GPUs} \times \text{sustained single-precision FLOPS per GPU}$$

Assumed GPU capacities: K80 = 2.8, K40 = 3.7, M40 = 6.0, **P100 = 9.5 TFLOPS**.

**Verify the base model:** $12 \text{ h} = 43{,}200$ s → $43{,}200 \times 8 \times 9.5 \times 10^{12} = 3.28 \times 10^{18}$ ≈ $3.3 \times 10^{18}$ ✓  
**Verify the big model:** $3.5$ days $= 302{,}400$ s → $302{,}400 \times 8 \times 9.5 \times 10^{12} = 2.30 \times 10^{19}$ ✓

**Comparison:** ConvS2S Ensemble used $7.7 \times 10^{19}$ FLOPs on EN-DE for 26.36 BLEU; the base Transformer used $3.3 \times 10^{18}$ — about **23× less compute** — for 27.3 BLEU.

---

## 20. Model Variations (Ablation Study, Table 3)

To evaluate the importance of each component, the base model was varied and measured on **English-German development set newstest2013**, using beam search but **no checkpoint averaging**. (Perplexities are per word-piece and are not comparable to per-word perplexities.)

**Base reference:** $N=6$, $d_{model}=512$, $d_{ff}=2048$, $h=8$, $d_k=d_v=64$, $P_{drop}=0.1$, $\epsilon_{ls}=0.1$, 100K steps → **PPL 4.92, BLEU 25.8, 65M params**.

### (A) Number of heads (keeping computation constant)

| $h$ | $d_k = d_v$ | PPL | BLEU |
|:---:|:---:|:---:|:---:|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| **8** | **64** | **4.92** | **25.8** |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

→ Single-head is **0.9 BLEU worse** than the best setting; but **too many heads also hurts** quality (each head becomes too small).

### (B) Key size $d_k$

| $d_k$ | PPL | BLEU | Params |
|:---:|:---:|:---:|:---:|
| 16 | 5.16 | 25.1 | 58M |
| 32 | 5.01 | 25.4 | 60M |

→ Reducing $d_k$ hurts. Determining compatibility is not easy; a more sophisticated compatibility function than dot product may be beneficial.

### (C) Model size

| Change | PPL | BLEU | Params |
|:---|:---:|:---:|:---:|
| $N = 2$ | 6.11 | 23.7 | 36M |
| $N = 4$ | 5.19 | 25.3 | 50M |
| $N = 8$ | 4.88 | 25.5 | 80M |
| $d_{model}=256$, $d_k=d_v=32$ | 5.75 | 24.5 | 28M |
| $d_{model}=1024$, $d_k=d_v=128$ | 4.66 | 26.0 | 168M |
| $d_{ff}=1024$ | 5.12 | 25.4 | 53M |
| $d_{ff}=4096$ | 4.75 | 26.2 | 90M |

→ **Bigger models are better.**

### (D) Dropout and label smoothing

| Change | PPL | BLEU |
|:---|:---:|:---:|
| $P_{drop} = 0.0$ | 5.77 | 24.6 |
| $P_{drop} = 0.2$ | 4.95 | 25.5 |
| $\epsilon_{ls} = 0.0$ | 4.67 | 25.3 |
| $\epsilon_{ls} = 0.2$ | 5.47 | 25.7 |

→ **Dropout is very helpful in avoiding overfitting** (removing it costs 1.2 BLEU).

### (E) Positional encoding

| Change | PPL | BLEU |
|:---|:---:|:---:|
| Learned positional embedding instead of sinusoids | 4.92 | 25.7 |

→ Nearly identical to the base model.

### Big model
$N=6$, $d_{model}=1024$, $d_{ff}=4096$, $h=16$, $P_{drop}=0.3$, 300K steps → **PPL 4.33, BLEU 26.4, 213M params**.

---

## 21. Generalization: English Constituency Parsing

To test whether the Transformer generalizes beyond translation, it was applied to **English constituency parsing** — converting a sentence into a syntax tree (as a linearized string).

```
Input:   "John saw the dog"
Output:  (S (NP John) (VP saw (NP the dog)))
```

**Why it's hard:** the output is subject to strong structural constraints and is **significantly longer than the input**. Also, RNN seq2seq models had not achieved SOTA in **small-data regimes**.

### Setup
- 4-layer Transformer, $d_{model} = 1024$.
- **WSJ only:** the Wall Street Journal portion of the Penn Treebank, ≈**40K training sentences**, vocabulary of **16K** tokens.
- **Semi-supervised:** additionally the high-confidence and BerkeleyParser corpora, ≈**17M sentences**, vocabulary of **32K** tokens.
- Only a small number of experiments to select dropout (both attention and residual), learning rates and beam size on the Section 22 development set; all other parameters unchanged from the EN-DE base model.
- Inference: max output length = **input length + 300**, **beam size 21**, $\alpha = 0.3$.

### Table 4: results on WSJ Section 23 (F1)

| Parser | Training | WSJ 23 F1 |
|:---|:---|:---:|
| Vinyals & Kaiser et al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| **Transformer (4 layers)** | **WSJ only, discriminative** | **91.3** |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) | semi-supervised | 92.1 |
| **Transformer (4 layers)** | **semi-supervised** | **92.7** |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

**Takeaways:**
- Despite little task-specific tuning, the Transformer performs **surprisingly well**, better than all previously reported models except the Recurrent Neural Network Grammar (Dyer et al. 2016).
- With only 40K WSJ sentences, it **outperforms the BerkeleyParser** (Petrov et al., 90.4 vs. 91.3) — unlike RNN seq2seq models (88.3).

---

## 22. Attention Visualizations (Interpretability)

The appendix shows attention patterns from the **encoder self-attention in layer 5 of 6**:

1. **Long-distance dependencies (Figure 3):** in *"It is in this spirit that a majority of American governments have passed new laws since 2009 **making** the registration or voting process more difficult."*, many heads attend from **"making"** to the distant word **"more difficult"**, completing the phrase *"making … more difficult"*. Different colors in the figure = different heads.

2. **Anaphora resolution (Figure 4):** in *"The Law will never be perfect, but **its** application should be just — this is what we are missing, in my opinion."*, heads 5 and 6 attend from **"its"** sharply to **"Law"** (and "application") — the model figured out what "its" refers to.

3. **Sentence structure (Figure 5):** different heads exhibit behavior clearly related to the **structure of the sentence** — the heads learned to perform **different tasks**.

```
Head 5, word "its":
   The   Law   will  never  be  perfect  ,  but  its  application ...
    ·    ███    ·     ·     ·     ·      ·   ·    ·       ██
         ↑ strong attention ("its" = "the Law's")
```

---

## 23. End-to-End Worked Example: Translating "I love cats"

Let's trace the whole pipeline with real (base model) shapes.

### Training (one example, parallel)

```
1. Source tokens:   ["I", "love", "cats"]                         → ids (3,)
2. Embed × 22.63 + PE                                              → (3, 512)
3. Dropout(0.1)                                                    → (3, 512)
4. 6 encoder layers:
      self-attn (8 heads × 64) → add & norm → FFN (512→2048→512) → add & norm
                                                                   → memory z (3, 512)
5. Target input (shifted right):  ["<s>", "Ich", "liebe", "Katzen"] → (4, 512) after embed+PE
6. 6 decoder layers:
      masked self-attn (4×4 scores, upper triangle = −∞) → add & norm
      cross-attn       (4×3 scores: German Q vs English K,V) → add & norm
      FFN → add & norm                                              → (4, 512)
7. Linear (shared embedding matrix) → logits                        → (4, 37000)
8. Softmax → probabilities                                          → (4, 37000)
9. Targets: ["Ich", "liebe", "Katzen", "</s>"] with label smoothing 0.1
10. Loss = average cross-entropy over the 4 positions → backprop → Adam step
```

All 4 target positions are predicted **in one forward pass** — that's the parallelism that RNNs lacked.

**Loss example:** if the model assigns probabilities 0.70, 0.60, 0.80, 0.90 to the correct tokens, the (unsmoothed) loss is
$-\frac{1}{4}(\ln 0.7 + \ln 0.6 + \ln 0.8 + \ln 0.9) = \frac{0.357 + 0.511 + 0.223 + 0.105}{4} = 0.299$,
and perplexity $= e^{0.299} \approx 1.35$.

### Inference (sequential decoding)

```
Encoder runs ONCE → memory z (3, 512)

Step 1: decoder input ["<s>"]                         → predicts "Ich"
Step 2: decoder input ["<s>", "Ich"]                  → predicts "liebe"
Step 3: decoder input ["<s>", "Ich", "liebe"]         → predicts "Katzen"
Step 4: decoder input ["<s>", "Ich", "liebe", "Katzen"] → predicts "</s>"  → STOP
(with beam size 4 and α = 0.6 in the real system; max length = 3 + 50 = 53)
```

> Training is parallel; **generation is still sequential** (one token per step). Making generation less sequential is listed by the authors as a future research goal.

---

## 24. Author Contributions

The paper lists authors in random order ("equal contribution"), with a footnote describing roles:

- **Jakob Uszkoreit** proposed replacing RNNs with self-attention and started the effort to evaluate this idea.
- **Ashish Vaswani**, with Illia Polosukhin, designed and implemented the first Transformer models and was crucially involved in every aspect of the work.
- **Noam Shazeer** proposed **scaled dot-product attention**, **multi-head attention** and the **parameter-free position representation**, and was involved in nearly every detail.
- **Niki Parmar** designed, implemented, tuned and evaluated countless model variants in the original codebase and tensor2tensor.
- **Llion Jones** experimented with novel model variants, was responsible for the initial codebase, and efficient inference and visualizations.
- **Łukasz Kaiser** and **Aidan N. Gomez** spent countless long days designing and implementing parts of **tensor2tensor**, replacing the earlier codebase, greatly improving results and accelerating research.

(Work performed while some authors were at Google Brain / Google Research.)

---

## 25. Limitations, Subtleties & Beginner FAQ

**Q1: Does the Transformer have *no* sense of order without PE?**  
Correct. Self-attention and FFN are **permutation-equivariant**: shuffle the input tokens and the outputs get shuffled the same way. Positional encodings are the *only* source of order.

**Q2: Is self-attention always cheaper than an RNN?**  
No. Cost is $O(n^2 d)$ — **quadratic in length**. For $n > d$ (e.g., long documents, $n = 2000$ vs. $d = 512$), it's more expensive (see Section 16). This motivated later work on efficient/sparse attention.

**Q3: What does "position-wise" mean in FFN?**  
The same 2-layer MLP is applied to each token vector independently. Token 1 and token 3 use identical weights, but don't interact inside the FFN.

**Q4: Why $-\infty$ and not just 0 for masking?**  
Masking happens **before** softmax. A score of 0 gives $e^0 = 1$ (still non-zero weight). $-\infty$ gives $e^{-\infty} = 0$ — truly blocked.

**Q5: Why is the decoder input "shifted right"?**  
So that at position $i$, the model sees tokens $< i$ and predicts token $i$. Without shifting, the model would see the token it is supposed to predict.

**Q6: Why does label smoothing make perplexity worse but BLEU better?**  
Perplexity rewards confident correct predictions; label smoothing deliberately makes the model less confident. But less overconfidence generalizes better and helps beam search, which improves BLEU.

**Q7: Are the 6 encoder layers sharing weights?**  
No. They are identical in **structure** but each has its **own** parameters.

**Q8: Is the decoder's cross-attention using the output of each corresponding encoder layer?**  
No. **Every** decoder layer attends to the **final** encoder output (layer 6).

**Q9: Why 8 heads of 64 rather than 1 head of 512?**  
Same cost (Section 8.5), but multiple heads capture different relations. Table 3 shows 1 head: 24.9 BLEU vs. 8 heads: 25.8 BLEU.

**Q10: What's the role of $W^O$?**  
After concatenating heads, $W^O$ mixes information across heads and projects back to $d_{model}$.

**Q11: Limitations acknowledged by the paper?**  
- Reduced effective resolution due to averaging attention-weighted positions (mitigated by multi-head).
- Quadratic cost for very long sequences → restricted attention suggested as future work.
- Output generation remains sequential.

---

## 26. Summary & Lasting Legacy

### The paper in 6 sentences
1. RNNs are sequential and struggle with long-range dependencies.
2. The Transformer replaces recurrence entirely with **multi-head self-attention** + position-wise **feed-forward** layers.
3. **Scaled dot-product attention** ($\text{softmax}(QK^T/\sqrt{d_k})V$) is fast and stable; **multi-head** attention lets the model attend to multiple relations simultaneously.
4. **Sinusoidal positional encodings** inject word order; **residual connections, layer norm, dropout, label smoothing** and a **warmup LR schedule** make training work.
5. Results: **28.4 BLEU** EN-DE and **41.8 BLEU** EN-FR — new state of the art with much less training compute (base model: 12 hours on 8 P100s).
6. It generalizes to other tasks such as constituency parsing (92.7 F1 semi-supervised).

### Future directions the authors proposed
- Extend the Transformer to problems with input/output modalities **other than text**: **images, audio and video**.
- Investigate **local, restricted attention** to efficiently handle large inputs/outputs.
- Make generation **less sequential**.

### What actually happened next 🚀

```
2017  Transformer ("Attention Is All You Need")
  │
  ├── 2018  GPT-1        (decoder-only Transformer, language modeling)
  ├── 2018  BERT         (encoder-only Transformer, masked LM)
  ├── 2019  GPT-2, T5, Transformer-XL
  ├── 2020  GPT-3, ViT   (Transformers for images — as the authors predicted!)
  ├── 2021  AlphaFold 2, CLIP, Whisper (audio)
  └── 2022+ ChatGPT, GPT-4, Gemini, Claude, LLaMA ... all Transformers
```

Practically every modern foundation model — for text, images, audio, video, protein structure and code — is built on the architecture introduced in this paper. **Attention really was (almost) all we needed.**
