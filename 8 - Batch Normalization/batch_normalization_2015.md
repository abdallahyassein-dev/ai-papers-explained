# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

> **Paper:** [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
> **Authors:** Sergey Ioffe, Christian Szegedy (Google)
> **Published:** February 2015 (ICML 2015)
> **Citations:** 40,000+ — One of the most influential deep learning papers ever written.

---

## Table of Contents

1. [Paper Overview (TL;DR)](#1-paper-overview-tldr)
2. [Key Terms & Definitions](#2-key-terms--definitions)
3. [The Problem: Internal Covariate Shift](#3-the-problem-internal-covariate-shift)
4. [Why Naive Normalization Fails](#4-why-naive-normalization-fails)
5. [The Solution: Batch Normalization](#5-the-solution-batch-normalization)
6. [The BN Algorithm Step-by-Step](#6-the-bn-algorithm-step-by-step)
7. [Learnable Parameters: γ and β](#7-learnable-parameters-γ-and-β)
8. [Training vs. Inference](#8-training-vs-inference)
9. [BN for Convolutional Networks](#9-bn-for-convolutional-networks)
10. [Why BN Enables Higher Learning Rates](#10-why-bn-enables-higher-learning-rates)
11. [BN as a Regularizer](#11-bn-as-a-regularizer)
12. [Backpropagation Through BN](#12-backpropagation-through-bn)
13. [Experiments & Results](#13-experiments--results)
14. [Practical Recipes When Using BN](#14-practical-recipes-when-using-bn)
15. [Summary of Contributions](#15-summary-of-contributions)

---

## 1. Paper Overview (TL;DR)

The core idea of this paper is surprisingly simple yet extremely powerful:

> **Normalize the inputs of each layer** so that they have zero mean and unit variance, computed over each mini-batch. Then add two learnable parameters (scale and shift) to preserve the network's representational power.

**Why it matters:**
- Training converges **14× faster** (same accuracy with 14× fewer training steps)
- Allows **much higher learning rates** without divergence
- Acts as a **regularizer**, reducing or eliminating the need for Dropout
- Makes it possible to train with **sigmoid activations** (previously impractical for deep nets)
- Achieved **state-of-the-art** on ImageNet (4.8% top-5 error), surpassing human-level accuracy

---

## 2. Key Terms & Definitions

Before diving in, let's define every important term used in this paper:

### Neural Network Basics

| Term | Definition | Example |
|------|-----------|---------|
| **SGD (Stochastic Gradient Descent)** | An optimization algorithm that updates model parameters using gradients computed on small random subsets (mini-batches) of data, rather than the full dataset. | Instead of computing the gradient on 1 million images, compute it on a batch of 32 images and update weights. |
| **Mini-batch** | A small subset of training examples used in one iteration of SGD. | If dataset has 60,000 images and batch size is 64, one epoch has ~937 mini-batches. |
| **Learning Rate** | A hyperparameter (α) that controls how big each parameter update step is. Too high → divergence. Too low → slow training. | α = 0.001 means each parameter changes by 0.001 × gradient. |
| **Activation Function** | A non-linear function applied after a linear transformation in each layer. Introduces non-linearity into the network. | Sigmoid: σ(x) = 1/(1+e⁻ˣ), ReLU: max(0, x) |
| **Saturating Nonlinearity** | An activation function whose output "plateaus" (gradient → 0) for large absolute input values. The neuron stops learning. | Sigmoid saturates near 0 and 1. Tanh saturates near -1 and 1. |
| **Vanishing Gradient** | When gradients become extremely small during backpropagation, causing early layers to learn very slowly or not at all. | With sigmoid, if x = 10, σ'(10) ≈ 0.0000454 — almost zero. |
| **Epoch** | One complete pass through the entire training dataset. | If you have 50,000 training images and batch size 100, one epoch = 500 steps. |

### Paper-Specific Terms

| Term | Definition | Analogy |
|------|-----------|---------|
| **Covariate Shift** | When the input distribution to a model changes between training and test time (or over time). | Imagine training a cat detector on indoor photos, then testing on outdoor photos — the "covariate" (input) has "shifted." |
| **Internal Covariate Shift (ICS)** | The change in the distribution of inputs to intermediate layers of a deep network, caused by parameter updates in preceding layers during training. | Layer 5 expects inputs with mean ≈ 2 and std ≈ 0.5, but after layer 3 updates its weights, layer 5 suddenly gets inputs with mean ≈ -1 and std ≈ 3. Layer 5 now has to re-adapt. |
| **Whitening** | Transforming data to have zero mean, unit variance, and zero correlation between features. Full whitening is expensive (requires computing a covariance matrix and its inverse square root). | Like standardizing test scores: subtract mean, divide by std, and decorrelate subjects. |
| **Normalization** | A simpler version of whitening: making data have zero mean and unit variance, without decorrelating. | Just subtract mean and divide by std, per feature independently. |
| **Population Statistics** | The mean and variance computed over the entire training dataset. Used at inference time. | The "true" mean and variance across all 1.2 million ImageNet images. |
| **Batch Statistics** | The mean and variance estimated from a single mini-batch. Used at training time. | Mean and variance across just 32 images in the current batch. |

### Math Notation Used in the Paper

| Symbol | Meaning |
|--------|---------|
| x | Input to a layer (a vector of activations) |
| x⁽ᵏ⁾ | The k-th dimension (feature) of input x |
| μ_B | Mean of activation values over a mini-batch B |
| σ²_B | Variance of activation values over a mini-batch B |
| x̂ | Normalized activation (zero mean, unit variance) |
| γ | Learnable scale parameter |
| β | Learnable shift parameter |
| y | Output of the BN transform: y = γx̂ + β |
| ε | Small constant for numerical stability (e.g., 1e-5) |
| m | Mini-batch size |
| W | Weight matrix of a layer |
| b | Bias vector of a layer |
| Θ | All trainable parameters of the network |
| ℓ | Loss function |

---

## 3. The Problem: Internal Covariate Shift

### The Core Issue

A deep neural network is a stack of layers. Each layer takes the output of the previous layer as input, applies a linear transformation, and passes the result through an activation function:

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → ... → [Layer N] → Output
           ↓            ↓            ↓                 ↓
        Updates      Updates      Updates           Updates
        weights      weights      weights           weights
```

**The problem:** When Layer 1 updates its weights, the distribution of its output changes. This means the input to Layer 2 changes. Layer 2 then has to adapt to this new input distribution. But when Layer 2 adapts (updates weights), the input to Layer 3 changes. And so on.

This cascading effect means **every layer is chasing a moving target** — the distribution of its input keeps shifting.

### Concrete Example

Imagine a 3-layer network processing images:

```
Step 1: Layer 2 receives inputs with mean = 2.0, std = 0.5
        → Layer 2 has learned weights that work well for this distribution

Step 2: Layer 1 updates its weights (gradient descent step)
        → Now Layer 2 receives inputs with mean = -0.3, std = 1.8
        → Layer 2's weights are now poorly calibrated
        → Layer 2 must "re-learn" to adapt

Step 3: Layer 2 updates its weights to adapt
        → Now Layer 3's inputs shift
        → Layer 3 must re-adapt

... This cascade repeats every single training step.
```

### Why This Is Especially Bad with Sigmoid

The sigmoid function σ(x) = 1/(1+e⁻ˣ) outputs values between 0 and 1:

```
            1.0 ─────────────────────────── (saturated: gradient ≈ 0)
               /
              /
             /   ← Linear regime (gradient is useful)
            /
           /
0.0 ──────/───────────────────────────────── (saturated: gradient ≈ 0)
       -6  -4  -2   0   2   4   6
```

When the input distribution shifts, many neurons' inputs can land in the **saturated regions** (|x| > 4), where the gradient is nearly zero. The network stops learning — this is the **vanishing gradient problem**.

The paper's insight: If we could **keep the input distribution stable** (zero mean, unit variance), we'd keep most activations in the **linear regime** of the sigmoid, where gradients flow well.

---

## 4. Why Naive Normalization Fails

### Attempt 1: Normalize Outside the Gradient Computation

A simple idea: after each gradient step, normalize the activations by subtracting the mean computed over the training set.

Consider a layer that computes:
```
x = u + b          (add bias)
x̂ = x - E[x]      (subtract mean to normalize)
```

**The problem:** Gradient descent doesn't know that `E[x]` depends on `b`. So it computes:

```
Δb ∝ -∂ℓ/∂x̂       (gradient update to bias)
```

After the update: `b ← b + Δb`

But then:
```
x̂_new = (u + b + Δb) - E[u + b + Δb]
       = u + b - E[u + b]
       = x̂_old                          ← Nothing changed!
```

The normalization **cancels out** the gradient update. The bias `b` keeps growing indefinitely while the output (and loss) remain unchanged. The authors observed this "blowing up" behavior empirically.

### Attempt 2: Full Whitening

We could do proper whitening (decorrelation + normalization) and compute its Jacobian for backpropagation. But this requires:

1. Computing the **covariance matrix** Cov[x] = E[xxᵀ] - E[x]E[x]ᵀ — this is a d×d matrix where d is the layer width
2. Computing its **inverse square root** Cov[x]⁻¹ᐟ²
3. Computing **derivatives** through these operations for backpropagation

For a layer with 4096 neurons, this means a 4096 × 4096 covariance matrix — **prohibitively expensive** to compute at every training step.

### The Solution Path

We need something that is:
- ✅ Differentiable (gradients can flow through it)
- ✅ Part of the computation graph (not applied externally)
- ✅ Cheap to compute
- ✅ Doesn't break the network's representational power

This leads to **Batch Normalization**.

---

## 5. The Solution: Batch Normalization

Batch Normalization makes **two key simplifications** compared to full whitening:

### Simplification 1: Normalize Each Feature Independently

Instead of full whitening (which decorrelates features), BN normalizes **each scalar feature independently**:

```
For a layer with d-dimensional input x = (x⁽¹⁾, x⁽²⁾, ..., x⁽ᵈ⁾):

  Normalize each x⁽ᵏ⁾ to have mean 0 and variance 1, independently.
```

This is much cheaper than computing a full covariance matrix. And as noted by LeCun et al. (1998), this still speeds up convergence even without decorrelation.

### Simplification 2: Use Mini-Batch Statistics Instead of Population Statistics

Instead of computing the mean and variance over the **entire training set** (expensive and impractical with SGD), BN uses the **current mini-batch** to estimate them:

```
Mini-batch B = {x₁, x₂, ..., xₘ}   (m examples)

μ_B = (1/m) Σᵢ xᵢ                   (batch mean)
σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²        (batch variance)
```

This works because:
- The mini-batch mean is an **unbiased estimator** of the population mean
- The mini-batch variance is a **reasonable estimator** of the population variance
- Both are **differentiable** — gradients can flow through them!

---

## 6. The BN Algorithm Step-by-Step

### Algorithm 1: Batch Normalizing Transform (Training)

```
Input:  Values of x over a mini-batch: B = {x₁, x₂, ..., xₘ}
        Learnable parameters: γ, β
        Small constant: ε (e.g., 1e-5)

Output: {yᵢ = BN_{γ,β}(xᵢ)}

Step 1: Compute mini-batch mean
        μ_B = (1/m) Σᵢ₌₁ᵐ xᵢ

Step 2: Compute mini-batch variance
        σ²_B = (1/m) Σᵢ₌₁ᵐ (xᵢ - μ_B)²

Step 3: Normalize
        x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

Step 4: Scale and shift
        yᵢ = γ · x̂ᵢ + β
```

### Numerical Example

Let's walk through a concrete example with a mini-batch of 4 values for a single neuron:

```
Mini-batch: B = {x₁=1, x₂=3, x₃=5, x₄=7},  ε = 0.001

Step 1: μ_B = (1+3+5+7) / 4 = 4.0

Step 2: σ²_B = [(1-4)² + (3-4)² + (5-4)² + (7-4)²] / 4
             = [9 + 1 + 1 + 9] / 4
             = 5.0

Step 3: x̂₁ = (1 - 4) / √(5 + 0.001) = -3 / 2.236 = -1.342
        x̂₂ = (3 - 4) / √(5 + 0.001) = -1 / 2.236 = -0.447
        x̂₃ = (5 - 4) / √(5 + 0.001) =  1 / 2.236 =  0.447
        x̂₄ = (7 - 4) / √(5 + 0.001) =  3 / 2.236 =  1.342

        Verify: mean(x̂) = (-1.342 + -0.447 + 0.447 + 1.342) / 4 ≈ 0  ✓
        Verify: var(x̂)  ≈ 1  ✓

Step 4: (Assuming γ=1, β=0 initially)
        y₁ = 1 × (-1.342) + 0 = -1.342
        y₂ = 1 × (-0.447) + 0 = -0.447
        y₃ = 1 × ( 0.447) + 0 =  0.447
        y₄ = 1 × ( 1.342) + 0 =  1.342
```

### Where BN Goes in the Network

The paper recommends placing BN **before the activation function** and **after the linear transformation**:

```
Without BN:  z = g(Wu + b)
With BN:     z = g(BN(Wu))         ← Note: bias b is removed (explained below)

Where:
  u = input from previous layer
  W = weight matrix
  g = activation function (sigmoid, ReLU, etc.)
  BN = Batch Normalization transform
```

**Why remove the bias b?** Because BN subtracts the mean (Step 1), which cancels out any constant added by the bias. The bias's role is now handled by the learnable parameter β in the BN transform.

---

## 7. Learnable Parameters: γ and β

### Why Do We Need Them?

Consider what happens if we only normalize (Steps 1-3, no γ and β):

- The normalized values x̂ are **always** constrained to mean 0, variance 1
- For a sigmoid activation, this means inputs are constrained to the **linear regime** around x = 0
- The sigmoid in its linear regime behaves approximately like a linear function
- We've effectively **removed the nonlinearity** from the network!

```
Sigmoid around x = 0:

σ(x) ≈ 0.5 + 0.25x    (approximately linear!)

If x̂ always has mean 0 and std 1, then:
  ~95% of values fall in [-2, 2]
  In this range, sigmoid is nearly linear
  → We lose the representational power of the nonlinearity
```

### The Identity Trick

The learnable parameters γ (scale) and β (shift) solve this:

```
y = γ · x̂ + β
```

**Key insight:** If the network learns that the optimal distribution is the **original unnormalized distribution**, it can recover it by setting:

```
γ = √(Var[x])    (= standard deviation of x)
β = E[x]         (= mean of x)

Then: y = √(Var[x]) · (x - E[x])/√(Var[x]) + E[x] = x
```

So BN with γ, β can represent the **identity transformation** — it can "undo" the normalization if that's what's optimal. The network decides how much normalization it actually needs.

### What the Network Actually Learns

In practice, γ and β rarely recover the original distribution. Instead, the network learns the optimal distribution for each activation:

```
Example of what might be learned:
  Neuron 1: γ = 0.8, β = 0.1   → Slightly compressed, slightly shifted
  Neuron 2: γ = 2.3, β = -0.5  → Expanded and shifted left
  Neuron 3: γ = 1.0, β = 0.0   → Kept normalized (normalization was optimal)
```

---

## 8. Training vs. Inference

### The Problem at Inference Time

During **training**, BN uses mini-batch statistics (μ_B, σ²_B). But during **inference** (deployment), we often process **one example at a time** — there is no mini-batch to compute statistics from.

Even with batches at inference time, we want the output to be **deterministic** — the same input should always produce the same output, regardless of what other examples happen to be in the batch.

### The Solution: Moving Averages

During training, BN keeps **running (moving) averages** of the batch statistics:

```
At each training step t:

  E[x]   ← momentum × E[x]   + (1 - momentum) × μ_B
  Var[x] ← momentum × Var[x] + (1 - momentum) × σ²_B

Typical momentum: 0.9 or 0.1 (framework dependent)
```

### Inference Computation

At inference time, the BN transform becomes a **simple linear transformation** (no dependence on other examples):

```
Inference:
  x̂ = (x - E[x]) / √(Var[x] + ε)     ← Uses population statistics
  y = γ · x̂ + β

This can be fused into a single linear transform:
  y = (γ / √(Var[x] + ε)) · x + (β - γ · E[x] / √(Var[x] + ε))
  y = W_bn · x + b_bn    ← Just a linear transform! Very fast.
```

> **Key point:** After training, BN adds zero computational overhead at inference time because it can be absorbed into the preceding linear layer.

### Algorithm 2: Training a Batch-Normalized Network (Summary)

```
Training:
1. For each mini-batch:
   a. Forward pass: compute BN using mini-batch statistics
   b. Backward pass: compute gradients through BN (see Section 12)
   c. Update parameters: W, γ, β using gradients
   d. Update running averages: E[x], Var[x]

Inference:
1. Replace all BN layers with fixed linear transforms using
   the running averages computed during training
2. Process each input independently (no batch dependency)
```

### Unbiased Variance Estimate

The paper notes an important detail: for inference, the population variance uses Bessel's correction:

```
Var[x] = m/(m-1) · E_B[σ²_B]
```

This is because the mini-batch variance σ²_B (dividing by m) is a **biased estimator** of the true variance. Multiplying by m/(m-1) gives the **unbiased estimate** (dividing by m-1, just like in statistics).

**Example:**
```
If batch size m = 32 and average batch variance = 5.0:
  Unbiased population variance = (32/31) × 5.0 = 5.16

The correction is small for large batches but matters for small ones.
```

---

## 9. BN for Convolutional Networks

### The Key Difference: Shared Statistics per Feature Map

In a fully-connected layer, each neuron gets its own γ and β. But in a convolutional layer, we want to preserve the **convolutional property**: the same filter should behave the same way regardless of spatial location.

```
Fully-connected layer with 256 neurons:
  → 256 pairs of (γ, β)

Convolutional layer with 64 filters, each producing 32×32 feature maps:
  → 64 pairs of (γ, β)   ← One per filter, NOT per spatial location
```

### How Statistics Are Computed

For a convolutional layer with mini-batch size m and feature maps of size p × q:

```
Feature map k:
  Collect ALL activations across:
    - All m examples in the mini-batch
    - All p × q spatial locations
  Total values: m × p × q

  μ_B⁽ᵏ⁾ = mean over all m·p·q values
  σ²_B⁽ᵏ⁾ = variance over all m·p·q values
```

**Example:**
```
Mini-batch size m = 32
Feature map size: 16 × 16
Number of filters: 64

For each of the 64 filters:
  Effective batch size = 32 × 16 × 16 = 8,192 values
  Compute mean and variance over these 8,192 values
  Apply normalization with one (γ, β) pair per filter

Total learnable parameters added by BN: 64 × 2 = 128
```

---

## 10. Why BN Enables Higher Learning Rates

### Problem Without BN

In a deep network, a small change in parameters can **amplify** as it propagates through layers:

```
Layer 1 weights change by δ
→ Layer 1 output changes by ~Wδ
→ Layer 2 output changes by ~W₂W₁δ
→ Layer N output changes by ~(∏Wᵢ)δ   ← Can be huge!
```

If the learning rate is too high, these amplified changes cause:
1. **Gradient explosion**: Gradients grow exponentially → parameters blow up
2. **Gradient vanishing**: Gradients shrink exponentially → no learning
3. **Saturation**: Activations enter saturated regime → vanishing gradients

### How BN Fixes This

**Insight 1: Scale Invariance**

BN makes the network insensitive to the scale of its parameters. Consider scaling a weight matrix by a constant `a`:

```
Without BN:
  BN(Wu) → normalized version of Wu
  BN(aWu) → normalized version of aWu → same as BN(Wu)!

Because:
  Mean of aWu = a × Mean of Wu
  Variance of aWu = a² × Variance of Wu
  Normalized: (aWu - a·μ) / (a·σ) = (Wu - μ) / σ   ← Same result!
```

This means:
- Gradient w.r.t. (aW) = (1/a) × Gradient w.r.t. W
- **Larger weights → smaller gradients** → self-stabilizing!
- No explosive parameter growth

**Insight 2: Jacobian Singular Values ≈ 1**

The paper conjectures (with analysis for the linear case) that BN helps keep the Jacobian's singular values close to 1:

```
If x̂ and ẑ both have unit covariance (thanks to BN):
  ẑ = F(x̂) ≈ J·x̂    (linear approximation)
  
  Cov[ẑ] = J · Cov[x̂] · Jᵀ = J · I · Jᵀ = J·Jᵀ
  
  Since Cov[ẑ] = I (due to normalization):
  J·Jᵀ = I
  → All singular values of J equal 1
  → Gradients neither explode nor vanish!
```

**Practical Result:**
```
Without BN: Safe learning rate ~ 0.0015
With BN:    Safe learning rate ~ 0.045 (30× higher!)
```

---

## 11. BN as a Regularizer

### What Is Regularization?

> **Regularization** is any technique that prevents overfitting — the model memorizing the training data instead of learning general patterns. Examples: Dropout, L2 weight decay, data augmentation.

### How BN Regularizes

During training, each example is normalized using the **mini-batch statistics**, which are **noisy estimates** of the true population statistics:

```
True normalization:     x̂ = (x - μ_pop) / σ_pop
BN normalization:       x̂ = (x - μ_batch) / σ_batch

The noise:  μ_batch ≈ μ_pop + noise
            σ_batch ≈ σ_pop + noise
```

This noise has several effects:

1. **The same input produces slightly different outputs** depending on what other examples are in the batch
2. **No training example is seen in isolation** — it's always in the context of other random examples
3. The noise acts like a **stochastic perturbation** similar to Dropout

### BN vs. Dropout

| Aspect | Dropout | Batch Normalization |
|--------|---------|-------------------|
| **Mechanism** | Randomly zeroes out neurons | Adds noise via batch statistics |
| **At inference** | Disabled (multiply by keep probability) | Uses fixed population statistics |
| **Hyperparameter** | Dropout rate (e.g., 0.5) | Batch size (affects noise level) |
| **Effect** | Forces redundant representations | Stabilizes input distributions + adds noise |

The paper found that **BN can replace Dropout** in many cases, and in fact removing Dropout from BN-networks improved results (because BN already provides sufficient regularization).

---

## 12. Backpropagation Through BN

For BN to work with gradient descent, we need to compute gradients through the normalization steps. Using the chain rule:

```
Given: ∂ℓ/∂yᵢ  (gradient from above)

We need: ∂ℓ/∂xᵢ  (gradient to pass below)
         ∂ℓ/∂γ   (gradient for γ)
         ∂ℓ/∂β   (gradient for β)
```

### Derivation (Step by Step)

```
Forward:
  μ_B = (1/m) Σ xᵢ
  σ²_B = (1/m) Σ (xᵢ - μ_B)²
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
  yᵢ = γ · x̂ᵢ + β

Backward:
  ∂ℓ/∂γ = Σᵢ (∂ℓ/∂yᵢ) · x̂ᵢ

  ∂ℓ/∂β = Σᵢ (∂ℓ/∂yᵢ)

  ∂ℓ/∂x̂ᵢ = (∂ℓ/∂yᵢ) · γ

  ∂ℓ/∂σ²_B = Σᵢ (∂ℓ/∂x̂ᵢ) · (xᵢ - μ_B) · (-1/2)(σ²_B + ε)^(-3/2)

  ∂ℓ/∂μ_B = Σᵢ (∂ℓ/∂x̂ᵢ) · (-1/√(σ²_B + ε))
           + (∂ℓ/∂σ²_B) · (1/m) Σᵢ (-2)(xᵢ - μ_B)

  ∂ℓ/∂xᵢ = (∂ℓ/∂x̂ᵢ) · (1/√(σ²_B + ε))
          + (∂ℓ/∂σ²_B) · (2/m)(xᵢ - μ_B)
          + (∂ℓ/∂μ_B) · (1/m)
```

**Key insight:** The gradients w.r.t. `xᵢ` depend on **all other examples in the batch** (through μ_B and σ²_B). This is what makes BN fundamentally different from per-example normalization — the gradient computation couples all examples in the batch.

---

## 13. Experiments & Results

### Experiment 1: MNIST (Proof of Concept)

**Setup:**
- Simple network: 28×28 input → 3 hidden layers (100 neurons each, sigmoid) → 10 outputs
- 50,000 training steps, batch size 60
- Compared: baseline vs. same network with BN before each sigmoid

**Results:**
- BN network achieved **higher test accuracy**
- BN network's internal activation distributions remained **stable** during training
- Baseline network's distributions **shifted dramatically**, confirming internal covariate shift

### Experiment 2: ImageNet Classification (Main Experiments)

The paper tested BN on a modified Inception (GoogLeNet) architecture:
- 13.6 million parameters
- Trained on ImageNet (1.2M images, 1000 classes)
- Mini-batch size: 32

**Models compared:**

| Model | Learning Rate | Key Modifications | Steps to 72.2% Acc | Max Accuracy |
|-------|--------------|-------------------|--------------------:|-------------:|
| **Inception** (baseline) | 0.0015 | None | 31.0 × 10⁶ | 72.2% |
| **BN-Baseline** | 0.0015 | +BN only | ~13.0 × 10⁶ | 72.7% |
| **BN-x5** | 0.0075 (5×) | +BN, +modifications | 2.1 × 10⁶ | 73.0% |
| **BN-x30** | 0.045 (30×) | +BN, +modifications | Slower start | **74.8%** |
| **BN-x5-Sigmoid** | 0.0075 | +BN, sigmoid instead of ReLU | — | 69.8% |
| **Inception-Sigmoid** | 0.0015 | sigmoid instead of ReLU | — | 0.1% (chance!) |

**Key findings:**

1. **14× fewer steps** to match baseline accuracy (BN-x5 vs Inception)
2. **30× higher learning rate** worked without divergence (BN-x30)
3. **Sigmoid became trainable** — without BN, sigmoid Inception was stuck at random chance (1/1000 = 0.1%). With BN, it reached 69.8%!
4. **Higher final accuracy** — BN-x30 reached 74.8% vs baseline's 72.2%

### Experiment 3: Ensemble (State-of-the-Art)

| Ensemble | Top-5 Validation Error | Top-5 Test Error |
|----------|----------------------:|------------------:|
| Previous SOTA (He et al., 2015) | 4.94% | — |
| **BN-Inception Ensemble (6 models)** | **4.9%** | **4.82%** |
| Estimated Human Accuracy | ~5.1% | — |

> **Batch Normalization enabled surpassing human-level performance** on ImageNet classification.

---

## 14. Practical Recipes When Using BN

The paper discovered several important practical guidelines when using BN:

### Modifications That Help

| Modification | Rationale | Effect |
|-------------|-----------|--------|
| **Increase learning rate** | BN prevents gradient explosion/vanishing | Up to 30× higher LR worked |
| **Remove Dropout** | BN already regularizes | Faster training, no extra overfitting |
| **Reduce L2 weight regularization** | BN reduces the need for weight decay | Reduced by 5× improved validation accuracy |
| **Accelerate learning rate decay** | Network trains faster, so decay should be faster | 6× faster decay schedule |
| **Remove Local Response Normalization (LRN)** | BN makes LRN redundant | Simplifies architecture |
| **Shuffle training data more thoroughly** | BN's regularization depends on batch composition | ~1% accuracy improvement |
| **Reduce photometric distortions** | Network trains faster, sees each image fewer times | Focus on "real" images |

### Key Implementation Details

```python
# Pseudocode for BN layer (PyTorch-style)

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = ones(num_features)      # Learnable scale
        self.beta = zeros(num_features)       # Learnable shift
        self.running_mean = zeros(num_features)  # For inference
        self.running_var = ones(num_features)    # For inference
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        if training:
            # Use batch statistics
            mu = mean(x, axis=0)         # Mean over batch
            var = var(x, axis=0)         # Variance over batch

            # Update running averages
            self.running_mean = (1 - self.momentum) * self.running_mean \
                              + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var \
                              + self.momentum * var
        else:
            # Use population statistics
            mu = self.running_mean
            var = self.running_var

        # Normalize
        x_hat = (x - mu) / sqrt(var + self.eps)

        # Scale and shift
        y = self.gamma * x_hat + self.beta
        return y
```

### Where to Place BN

```
Recommended (paper's approach):
  Input → Linear(W) → BN → Activation → ...

Alternative (also used in practice):
  Input → Linear(W) → Activation → BN → ...

For ConvNets:
  Input → Conv(W, no bias) → BN → ReLU → ...
         ↑                    ↑
    Remove bias b         One (γ, β) per filter
```

---

## 15. Summary of Contributions

### What the Paper Introduced

1. **The concept of Internal Covariate Shift** — named and formalized the problem of shifting input distributions in deep networks

2. **Batch Normalization** — a practical, differentiable normalization technique that:
   - Normalizes each feature to zero mean and unit variance using mini-batch statistics
   - Adds learnable scale (γ) and shift (β) to preserve representational power
   - Uses running averages for deterministic inference
   - Adds only **2 parameters per activation** (minimal overhead)

3. **Empirical findings:**
   - 14× training speedup
   - 30× higher learning rates
   - Replaces Dropout
   - Enables sigmoid training in deep nets
   - State-of-the-art on ImageNet (surpassing human accuracy)

### Impact on the Field

Batch Normalization became **one of the most widely used techniques in deep learning**:

- Used in virtually **every modern architecture**: ResNet, DenseNet, EfficientNet, Transformers, GANs, etc.
- Spawned a family of normalization techniques:
  - **Layer Normalization** (Ba et al., 2016) — normalizes across features, used in Transformers
  - **Instance Normalization** (Ulyanov et al., 2016) — per-instance, per-channel, used in style transfer
  - **Group Normalization** (Wu & He, 2018) — compromise between LN and IN
  - **Weight Normalization** (Salimans & Kingma, 2016) — normalizes weight vectors
  - **Spectral Normalization** (Miyato et al., 2018) — constrains Lipschitz constant of layers

### Modern Perspective

While the paper attributed BN's success to reducing Internal Covariate Shift, later research (notably Santurkar et al., 2018 — "How Does Batch Normalization Help Optimization?") suggested that:
- BN's main benefit may be **smoothing the optimization landscape** rather than reducing ICS
- BN makes the loss surface significantly **smoother** (more Lipschitz continuous)
- This smoother landscape allows larger learning rates and faster convergence

Regardless of the exact mechanism, the **practical benefits of BN are undisputed** and it remains a foundational technique in deep learning.

---

## Appendix: Architecture Details

The Inception variant used in the paper (modified GoogLeNet) made these changes:
- Replaced 5×5 convolutions with **two consecutive 3×3 convolutions** (+9 weight layers, +25% parameters, +30% compute)
- Increased 28×28 inception modules from 2 to 3
- Mixed average and max pooling inside modules
- Used stride-2 convolution/pooling before filter concatenation (instead of separate pooling layers)
- Used separable convolution with depth multiplier 8 on the first conv layer
- Total: **13.6 million parameters**, no fully-connected layers (except final softmax)
