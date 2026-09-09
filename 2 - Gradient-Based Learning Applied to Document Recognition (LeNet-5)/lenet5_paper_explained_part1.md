# 📄 Gradient-Based Learning Applied to Document Recognition — The Complete Guide

> **Paper**: "Gradient-Based Learning Applied to Document Recognition"
> **Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
> **Published**: Proceedings of the IEEE, Vol. 86, No. 11, November 1998
> **Pages**: 46 pages (2278–2324)

> [!NOTE]
> **Your Background**: You've read "Learning representations by back-propagating errors" (Rumelhart, Hinton, Williams, 1986). That paper taught you how a neural network can learn by propagating errors backward through layers to adjust weights. This paper takes that foundation and builds an **entire system** for reading documents — from raw pixels to final answers. Everything here connects back to backpropagation.

---

# Table of Contents

1. [Section I: Introduction — The Big Picture](#section-i-introduction--the-big-picture)
2. [Section II: Convolutional Neural Networks](#section-ii-convolutional-neural-networks)
3. [Section III: LeNet-5 — The Full Architecture](#section-iii-lenet-5--the-full-architecture)
4. [Section IV: Loss Functions and Training](#section-iv-loss-functions-and-training)
5. [Section V: Graph Transformer Networks](#section-v-graph-transformer-networks)
6. [Section VI: The MNIST Experiments](#section-vi-the-mnist-experiments)
7. [Section VII: Multi-Module Systems and Check Reading](#section-vii-multi-module-systems-and-check-reading)
8. [Section VIII: Conclusions](#section-viii-conclusions)

---

# Section I: Introduction — The Big Picture

## What Problem Is This Paper Solving?

Imagine you work at a bank. Every day, **millions** of checks arrive. Each check has a handwritten dollar amount (like "$1,234.56"). You need to read that amount and put it into a computer. In 1998, **humans** did most of this work. It was slow, expensive, and error-prone.

The paper asks: **Can we build a machine that reads handwritten characters directly from images, automatically, with very few errors?**

### Why Was This Hard?

Let's say you want to recognize the digit "7". Simple idea: store a picture of a perfect "7" and compare every new image to it. But here's the problem:

```
Different ways people write "7":

    7     7̶     7     𝟕     ⁷
  (plain) (crossed) (slanted) (bold) (tiny)
```

Every person writes differently. The same person writes differently each time. The digit can be:
- **Shifted** left or right in the image
- **Rotated** slightly
- **Scaled** (bigger or smaller)
- **Distorted** (shaky hand, bad pen)
- Written with **different stroke thickness**
- On a **noisy background** (smudges, lines from the check form)

So a pixel-by-pixel comparison will **never** work reliably.

## The Old Way: Hand-Engineered Feature Extraction

Before this paper, the standard approach had **two separate steps**:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────┐
│   Raw Image      │────▶│  Feature          │────▶│ Classifier│────▶ Answer
│   (pixels)       │     │  Extractor        │     │           │
│                  │     │  (HAND-DESIGNED)   │     │ (trained) │
└──────────────────┘     └──────────────────┘     └──────────┘
```

**Step 1 — Feature Extractor (Hand-Designed):** A human expert would sit down and design rules to extract "features" from the image. For example:
- "Count the number of loops" (8 has two loops, 0 has one, 1 has none)
- "Measure the aspect ratio" (1 is tall and thin, 0 is round)
- "Look at which quadrant has the most ink"
- "Find endpoints and intersections of strokes"

**Step 2 — Classifier (Trained):** These features were fed into a simple classifier (like a small neural network) that was trained to map features to categories.

### The Big Problem With This Approach

The feature extractor was **not learned from data** — it was designed by a human. This had massive downsides:

1. **Expertise required**: You needed a domain expert who deeply understood handwriting
2. **Slow iteration**: Changing the feature extractor meant months of engineering work
3. **Not transferable**: Features designed for digits wouldn't work for letters, or for Chinese characters
4. **Suboptimal**: Human-designed features might miss patterns that a learning algorithm could find
5. **Fragile**: The features might work for clean images but fail on noisy real-world data

## The Paper's Revolutionary Idea

**Instead of hand-designing the feature extractor, LEARN it from data.**

```
┌──────────────────┐     ┌──────────────────────────────────┐     ┌──────────┐
│   Raw Image      │────▶│  ENTIRE SYSTEM IS LEARNED        │────▶│  Answer  │
│   (pixels)       │     │  Feature Extraction + Classifier  │     │          │
│                  │     │  ALL trained by backpropagation   │     │          │
└──────────────────┘     └──────────────────────────────────┘     └──────────┘
```

This is the core thesis: **Feed raw pixels in, get the answer out, and train EVERYTHING in between using gradient-based learning (i.e., backpropagation — the thing you already know!).**

### Connection to Your Backpropagation Knowledge

Remember from Rumelhart et al. (1986):
- You have a network with weights
- You feed input, get output
- You compare output to the desired answer using a **loss function**
- You compute the **gradient** of the loss with respect to each weight (using the chain rule)
- You adjust weights in the direction that reduces the loss

This paper says: "We can do **exactly** this, but on a much bigger, specially-designed network that takes entire images as input." The key innovation is the **architecture** of the network — which is what the rest of the paper describes.

---

## The Concept of "Learning Machine" (Formalized)

The paper formalizes the entire system as a **function**:

```
Yp = F(Zp, W)
```

Where:
- **Zp** = the p-th input pattern (e.g., a 32×32 pixel image of a digit)
- **W** = the set of all tunable parameters (weights) in the system
- **F** = the function computed by the network
- **Yp** = the output (e.g., a vector of 10 numbers, one for each digit 0-9)

### Example

Let's say the input `Zp` is an image of the digit "3":

```
Input Image Zp (simplified 5×5):     Output Yp (10 values):
                                     
  ■ ■ ■ □ □                          Y[0] = 0.01  (probability it's "0")
  □ □ ■ □ □                          Y[1] = 0.02  (probability it's "1")
  □ ■ ■ □ □                          Y[2] = 0.05  (probability it's "2")
  □ □ ■ □ □                          Y[3] = 0.89  (probability it's "3") ← HIGHEST
  ■ ■ ■ □ □                          Y[4] = 0.01  (probability it's "4")
                                     Y[5] = 0.01  ...
                                     Y[6] = 0.00
                                     Y[7] = 0.01
                                     Y[8] = 0.00
                                     Y[9] = 0.00
```

The desired output `Dp` for "3" would be `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

The **loss** measures how far `Yp` is from `Dp`. Backpropagation computes how to adjust `W` to make `Yp` closer to `Dp`.

---

## The Training Process

The paper describes the standard training process (which you know from backpropagation) but formalizes it:

### Loss Function

The average loss over a training set of P patterns:

```
E(W) = (1/P) × Σ(p=1 to P) Ep(Dp, F(Zp, W))
```

Where:
- `Ep` is the loss for a single pattern p
- `Dp` is the desired (correct) output for pattern p
- `F(Zp, W)` is what the network actually outputs for input Zp

### Gradient Descent (What You Already Know)

To minimize E(W), update each weight:

```
W(new) = W(old) - η × ∂E/∂W
```

Where:
- `η` (eta) = the **learning rate** (a small positive number like 0.01)
- `∂E/∂W` = the gradient (partial derivative of loss with respect to weights)

This is exactly what Rumelhart et al. taught you, but now applied to **much larger** networks.

### Stochastic Gradient Descent (SGD)

The paper introduces an important practical detail: **you don't need to compute the gradient over ALL training examples before updating weights**. Instead:

1. Pick a **single** training example (or a small batch)
2. Compute the gradient for just that example
3. Update the weights
4. Pick another example and repeat

```
W(new) = W(old) - η × ∂Ep/∂W
```

Notice: `∂Ep/∂W` (gradient from one example) instead of `∂E/∂W` (gradient from all examples).

#### Why SGD Works Better in Practice

**Example**: You have 60,000 training images.

**Full Gradient Descent**: 
- Look at all 60,000 images → compute one gradient → make one weight update
- This is like reading an entire textbook before correcting a single misconception

**Stochastic Gradient Descent**:
- Look at image #1 → compute gradient → update weights
- Look at image #2 → compute gradient → update weights
- ...
- After seeing 60,000 images, you've made 60,000 updates!

SGD is **much faster** to converge because:
1. The updates happen much more frequently
2. The "noise" in the gradient (from using one example) actually helps escape local minima
3. It introduces a form of implicit regularization

The paper notes that SGD with careful learning rate scheduling was the standard way to train these networks.

---

## Generalization: The Real Goal

The paper emphasizes that training accuracy is **not** the goal. What matters is **test accuracy** — how well the system performs on images it has **never seen during training**.

### The Bias-Variance Tradeoff

The paper discusses this critical concept:

**Underfitting (High Bias)**: The model is too simple to capture the patterns.
```
Example: Using a model that only looks at the center pixel.
Training accuracy: 20%
Test accuracy: 19%
Both are bad → model is too simple
```

**Overfitting (High Variance)**: The model memorizes the training data.
```
Example: A model with millions of free parameters on a small dataset.
Training accuracy: 99.9%
Test accuracy: 70%
Big gap → model memorized training data, doesn't generalize
```

**Good Fit**: The model captures real patterns without memorizing noise.
```
Training accuracy: 99%
Test accuracy: 98%
Small gap → model learned generalizable patterns
```

### How to Achieve Good Generalization

The paper discusses three approaches:

1. **Structural**: Design the architecture so it has an appropriate number of parameters (not too many, not too few). **This is what LeNet-5 does with weight sharing!**

2. **Regularization**: Add a penalty to the loss function that discourages large weights:
   ```
   E_total = E_data + λ × ||W||²
   ```
   This is like saying "not only should the output be correct, but the weights should be small."

3. **Collect more data**: More training examples help the model learn better patterns.

---

# Section II: Convolutional Neural Networks

This is the **heart** of the paper. To understand why convolutional neural networks (CNNs) were invented, you first need to understand why regular (fully connected) neural networks **fail** at image recognition.

## Why Fully Connected Networks Fail for Images

### The Curse of Dimensionality

Consider a 32×32 grayscale image. That's 32 × 32 = **1,024 pixels**. Each pixel is an input.

If you build a fully connected network with, say, 100 hidden neurons in the first layer:

```
Connections from input to first hidden layer:
  1,024 inputs × 100 neurons = 102,400 weights

And this is just ONE hidden layer!
```

For a more realistic image (like 256×256), you'd have 65,536 inputs:
```
65,536 × 100 = 6,553,600 weights — just for the first layer!
```

**Problems**:
1. **Too many parameters** → needs massive amounts of training data to avoid overfitting
2. **No spatial structure** → the network treats pixel (0,0) the same as pixel (31,31)
3. **Not shift-invariant** → if you move the digit 1 pixel to the right, ALL the learned weights become useless

### A Concrete Example of Why This Is Bad

```
Training example:        Test example (shifted 2 pixels right):

□ □ □ □ □ □ □ □          □ □ □ □ □ □ □ □
□ □ ■ □ □ □ □ □          □ □ □ □ ■ □ □ □
□ ■ □ ■ □ □ □ □          □ □ □ ■ □ ■ □ □
□ ■ ■ ■ □ □ □ □          □ □ □ ■ ■ ■ □ □
□ ■ □ ■ □ □ □ □          □ □ □ ■ □ ■ □ □
□ □ □ □ □ □ □ □          □ □ □ □ □ □ □ □
```

Both are clearly "A", but in a fully connected network, the **pixels that are "on"** are in completely different positions. The network trained on the left "A" has no idea what to do with the right "A" because it learned specific weights for specific pixel positions.

## The Three Key Ideas Behind CNNs

The paper introduces three architectural ideas that solve these problems:

### Idea 1: Local Receptive Fields

Instead of connecting every input pixel to every neuron, each neuron only looks at a **small local patch** of the image.

```
                          Full Image (7×7)
                        ┌─────────────────┐
                        │ . . . . . . .   │
                        │ . ┌─────┐ . .   │    ← This 3×3 region is the
                        │ . │ x x x│ . .   │      "receptive field" of
                        │ . │ x x x│ . .   │      ONE neuron
                        │ . │ x x x│ . .   │
                        │ . └─────┘ . .   │
                        │ . . . . . . .   │
                        └─────────────────┘
```

**Why this makes sense**: In images, pixels that are close together are **correlated** — they form edges, curves, textures. A pixel in the top-left corner tells you almost nothing about a pixel in the bottom-right corner. So it makes sense to only look at local neighborhoods.

**Example**: When you see a curve like the top of a "3", you see it by looking at a small local region. You don't need to see the entire image to detect that curve.

```
Local feature detection:

  □ □ ■         This small 3×3 patch contains a
  □ ■ □         diagonal edge going from top-right
  ■ □ □         to bottom-left.

A neuron looking at this patch can learn to detect "diagonal edge."
```

### Idea 2: Shared Weights (Weight Sharing)

Here's the breakthrough: **the same feature can appear anywhere in the image.** A horizontal edge in the top-left corner looks the same as a horizontal edge in the bottom-right corner. So why learn separate detectors for every position?

**Weight sharing** means: one set of weights (one "filter" or "kernel") is **slid across the entire image**, computing a result at each position.

```
Kernel (3×3):            Applied to position (0,0):    Applied to position (0,1):
                         
┌─────────┐              Image patch:     Result:      Image patch:     Result:
│ 1  0 -1 │              □ □ ■            sum =        □ ■ □            sum =
│ 1  0 -1 │              □ ■ □            Σ(kernel ×   ■ □ □            Σ(kernel ×
│ 1  0 -1 │              ■ □ □            patch)       □ □ □            patch)
└─────────┘              
```

The **same** kernel slides to every position. This gives us:

1. **Far fewer parameters**: Instead of 1,024 × 100 = 102,400 weights, you have one kernel of size 5×5 = 25 weights (plus 1 bias = 26 parameters for one feature map). Even with 6 feature maps, that's only 156 parameters vs. 102,400!

2. **Built-in shift invariance**: Because the same kernel is applied everywhere, it will detect a feature regardless of where it appears in the image.

The output of sliding one kernel across the entire image is called a **feature map**:

```
Input Image (7×7)          Kernel (3×3)          Feature Map (5×5)
                           ┌───────┐
┌─────────────┐            │ 1 0 -1│            ┌───────────┐
│ . . . . . . .│            │ 1 0 -1│            │ r r r r r │
│ . . . . . . .│     ×      │ 1 0 -1│     =      │ r r r r r │
│ . . . . . . .│            └───────┘            │ r r r r r │
│ . . . . . . .│                                 │ r r r r r │
│ . . . . . . .│                                 │ r r r r r │
│ . . . . . . .│                                 └───────────┘
│ . . . . . . .│
└─────────────┘
Each 'r' is the result of applying the kernel to
a 3×3 region of the input centered at that position.
```

You use **multiple kernels** to detect **different features** (horizontal edges, vertical edges, curves, etc.), producing multiple feature maps.

### Idea 3: Sub-Sampling (Pooling)

Once you've detected a feature (say, a horizontal edge), its **exact position** becomes less important. What matters is the **approximate position** relative to other features.

Sub-sampling reduces the spatial resolution, making the representation more compact and slightly invariant to small shifts and distortions.

```
Feature Map (4×4):         After 2×2 Sub-Sampling (2×2):

┌───────────┐              ┌─────┐
│ 1  3  2  4│              │ 2  3│    ← Each cell is the AVERAGE
│ 2  1  4  2│     ───▶     │ 3  2│      of the 2×2 block above it
│ 5  3  1  2│              └─────┘
│ 1  3  3  1│
└───────────┘

Block (1,3,2,1): average = (1+3+2+1)/4 = 1.75 ≈ 2
Block (2,4,4,2): average = (2+4+4+2)/4 = 3
Block (5,3,1,3): average = (5+3+1+3)/4 = 3
Block (1,2,3,1): average = (1+2+3+1)/4 = 1.75 ≈ 2
```

> [!IMPORTANT]
> In the original LeNet-5, sub-sampling is NOT simply averaging. Each sub-sampling unit computes: `output = sigmoid(w × (sum of 4 inputs) + b)`, where `w` (weight) and `b` (bias) are **trainable parameters**. So even the pooling operation is learned!

### The Overall CNN Pattern: Convolution → Sub-Sampling → Convolution → Sub-Sampling → ...

```
Input     Conv      Pool      Conv      Pool      Fully Connected
Image  → Layer  →  Layer  →  Layer  →  Layer  → Layer(s) → Output

32×32    28×28     14×14     10×10      5×5      120 → 84 → 10
         ×6        ×6        ×16       ×16

Resolution: HIGH ──────────────────────────────────▶ LOW
Features:   SIMPLE (edges) ─────────────────────────▶ COMPLEX (shapes)
```

Each pair of (convolution + sub-sampling) layers:
- **Increases** the semantic level of features (from edges to shapes to parts to whole objects)
- **Decreases** the spatial resolution (from 32×32 down to 5×5)
- **Increases** the number of feature maps (from 1 to 6 to 16)

**Analogy**: Think of how **you** recognize a digit:
1. First, your eyes detect low-level features: "There are edges here, curves there"
2. Then your brain combines edges into shapes: "There's a loop, there's a vertical stroke"
3. Then it combines shapes into parts: "There's a loop on top connected to a loop on bottom"
4. Finally it recognizes the whole thing: "That's an 8!"

A CNN does exactly this, but automatically learns what features to detect at each level.

---

# Section III: LeNet-5 — The Full Architecture

Now let's go through LeNet-5 **layer by layer**, with full detail. This is the specific architecture the paper proposes for handwritten digit recognition.

## Input: 32×32 Grayscale Image

The input is a 32×32 pixel grayscale image. The pixel values are normalized so that:
- The **background** (white) = -0.1
- The **foreground** (black, where the ink is) = 1.175

> [!NOTE]
> **Why 32×32 and not 28×28?** The MNIST digits are actually 20×20 pixels, centered in a 28×28 field. LeNet-5 pads this to 32×32 so that potential distinctive features (like stroke endpoints or corners) can appear in the **center** of a 5×5 receptive field even when they're near the edge of the digit. If the image were only 28×28, features near the border would fall at the edge of a receptive field and might not be detected properly.
>
> **Why these specific values (-0.1 and 1.175)?** This normalization ensures the mean input is approximately zero and the variance is approximately one. This helps gradient descent converge faster — a detail you'd appreciate from your backpropagation knowledge. If inputs had a mean far from zero, the gradients would be biased and training would be slower.

### Visualization

```
    32×32 image of "4":
    
    ┌────────────────────────────────┐
    │ . . . . . . . . . . . . . . . │  (padding zone: -0.1)
    │ . . . . . . . . . . . . . . . │
    │ . . . . . . . . . . . . . . . │
    │ . . . . . ■ . . . . . . . . . │  ← digit appears roughly centered
    │ . . . . ■ ■ . . . . . . . . . │
    │ . . . ■ . ■ . . . . . . . . . │
    │ . . ■ . . ■ . . . . . . . . . │
    │ . . ■ ■ ■ ■ ■ . . . . . . . . │  ← horizontal bar of "4"
    │ . . . . . ■ . . . . . . . . . │
    │ . . . . . ■ . . . . . . . . . │
    │ . . . . . ■ . . . . . . . . . │
    │ . . . . . . . . . . . . . . . │
    │ . . . . . . . . . . . . . . . │
    └────────────────────────────────┘
    
    ■ = 1.175 (ink)
    . = -0.1  (background)
```

---

## Layer C1: First Convolutional Layer

### Specifications
- **Input**: 32×32×1 (single-channel grayscale image)
- **Kernel size**: 5×5
- **Number of kernels (feature maps)**: 6
- **Stride**: 1 (the kernel moves 1 pixel at a time)
- **Padding**: None (valid convolution)
- **Output size**: 28×28×6
- **Activation function**: Scaled hyperbolic tangent: `f(x) = A × tanh(S × x)` where A = 1.7159, S = 2/3
- **Trainable parameters**: 6 × (5×5 + 1) = 6 × 26 = **156**
- **Connections**: 6 × 28 × 28 × 26 = **122,304**

### How the Output Size Is Calculated

```
Output size = (Input size - Kernel size) / Stride + 1
            = (32 - 5) / 1 + 1
            = 28
```

### What Happens (Step by Step)

Each of the 6 kernels slides across the entire 32×32 image:

```
Kernel #1 (detects, say, horizontal edges):
┌──────────────┐
│ -1 -1 -1 -1 -1│
│  0  0  0  0  0│
│  0  0  0  0  0│
│  0  0  0  0  0│
│  1  1  1  1  1│
└──────────────┘

Kernel #2 (detects, say, vertical edges):
┌──────────────┐
│ -1  0  0  0  1│
│ -1  0  0  0  1│
│ -1  0  0  0  1│
│ -1  0  0  0  1│
│ -1  0  0  0  1│
└──────────────┘

... and so on for kernels #3 through #6
```

> [!NOTE]
> These specific kernel values are just examples to build intuition! In reality, the network **learns** the kernel values through backpropagation. You don't manually specify them. The learning algorithm discovers that detecting horizontal edges, vertical edges, curves, etc. is useful for recognizing digits.

### The Convolution Operation in Detail

Let me show you exactly how one output pixel is computed:

```
For Kernel #1, computing the output at position (0,0):

Take the 5×5 patch of the input starting at (0,0):
Input patch:          Kernel #1:           Element-wise multiply:
┌────────────┐       ┌────────────┐       ┌──────────────────┐
│-.1 -.1 -.1 -.1 -.1│ × │ k00 k01 k02 k03 k04│ = │-.1×k00 -.1×k01 ...│
│-.1 -.1 -.1 -.1 -.1│   │ k10 k11 k12 k13 k14│   │-.1×k10 -.1×k11 ...│
│-.1 -.1 -.1 -.1 -.1│   │ k20 k21 k22 k23 k24│   │...               │
│-.1 -.1 -.1 -.1 -.1│   │ k30 k31 k32 k33 k34│   │...               │
│-.1 -.1 -.1 -.1 -.1│   │ k40 k41 k42 k43 k44│   │...               │
└────────────┘       └────────────┘       └──────────────────┘

Sum all 25 products + bias:
output(0,0) = Σ(input_patch × kernel) + bias_1

Then apply the activation function:
feature_map_1(0,0) = 1.7159 × tanh(2/3 × output(0,0))
```

Then slide the kernel one pixel to the right and compute output(0,1), and so on until you've covered all 28×28 positions.

### About the Activation Function

The paper uses a **scaled tanh** activation function:

```
f(x) = A × tanh(S × x)

where A = 1.7159 and S = 2/3
```

**Why not simple tanh?** The scaling constants are chosen so that:
- `f(1) ≈ 1` and `f(-1) ≈ -1` (nice, symmetric range)
- The second derivative (how curved the function is) is maximum at `x = 1`, which means the learning speed is highest when the output is near ±1 rather than near 0

**Graph of scaled tanh**:
```
Output
  1.7 ─┐                            ╭───────────
       │                         ╭──╯
       │                      ╭─╯
       │                   ╭─╯
  0.0 ─┤─────────────────×───────────────
       │               ╭─╯
       │            ╭─╯
       │         ╭──╯
 -1.7 ─┤────────╯
       └──┬──────┬──────┬──────┬──────┬──
         -4    -2      0      2      4   Input
```

This is different from the sigmoid you learned in the backpropagation paper! The sigmoid outputs [0, 1], but **tanh outputs [-1, 1]**, which is better because:
- Outputs are zero-centered → gradients don't have a systematic bias
- Negative values can represent "absence" of a feature

### What C1 Learns (Conceptually)

After training, the 6 kernels in C1 typically learn to detect:

```
Feature Map 1: Horizontal edges    Feature Map 2: Vertical edges
┌─────┐                           ┌─────┐
│─────│                           │ | | │
│     │                           │ | | │
│─────│                           │ | | │
└─────┘                           └─────┘

Feature Map 3: Diagonal (/)       Feature Map 4: Diagonal (\)
┌─────┐                           ┌─────┐
│    /│                           │\    │
│   / │                           │ \   │
│  /  │                           │  \  │
└─────┘                           └─────┘

Feature Map 5: Center spot         Feature Map 6: Edge/corner
┌─────┐                           ┌─────┐
│  .  │                           │■    │
│ ... │                           │■    │
│  .  │                           │     │
└─────┘                           └─────┘
```

---

## Layer S2: First Sub-Sampling (Pooling) Layer

### Specifications
- **Input**: 28×28×6 (6 feature maps from C1)
- **Pooling region**: 2×2
- **Stride**: 2 (non-overlapping pools)
- **Output size**: 14×14×6
- **Trainable parameters**: 6 × 2 = **12** (one weight and one bias per feature map)
- **Connections**: 6 × 14 × 14 × 5 = **5,880**

### How It Works

For each 2×2 block in a feature map:
1. **Sum** the 4 pixel values
2. **Multiply** by a trainable weight `w`
3. **Add** a trainable bias `b`
4. Pass through the **activation function**

```
Feature Map from C1 (showing a 4×4 region):

┌──────────┐
│ 0.5  0.8 │ 0.2  0.1│
│ 0.3  0.6 │ 0.4  0.2│
│──────────┼──────────│
│ 0.1  0.0 │ 0.9  0.7│
│ 0.2  0.1 │ 0.8  0.6│
└──────────┘

Pool 1 (top-left 2×2):  sum = 0.5+0.8+0.3+0.6 = 2.2
                         output = f(w × 2.2 + b)

Pool 2 (top-right 2×2): sum = 0.2+0.1+0.4+0.2 = 0.9
                         output = f(w × 0.9 + b)

Pool 3 (bottom-left 2×2): sum = 0.1+0.0+0.2+0.1 = 0.4
                           output = f(w × 0.4 + b)

Pool 4 (bottom-right 2×2): sum = 0.9+0.7+0.8+0.6 = 3.0
                            output = f(w × 3.0 + b)

Result (2×2 from this 4×4 region):
┌─────────┐
│ f(w×2.2+b)  f(w×0.9+b)│
│ f(w×0.4+b)  f(w×3.0+b)│
└─────────┘
```

### Why Sub-Sampling?

1. **Reduces computation**: 28×28 → 14×14 means 4× fewer values per feature map
2. **Adds robustness**: If the digit shifts by 1 pixel, the sub-sampled representation barely changes
3. **Forces the network to learn higher-level features**: By shrinking the spatial dimensions, the next convolutional layer is forced to combine features from a larger area of the original image

---

## Layer C3: Second Convolutional Layer

### Specifications
- **Input**: 14×14×6 (6 feature maps from S2)
- **Kernel size**: 5×5
- **Number of feature maps**: 16
- **Output size**: 10×10×16
- **Trainable parameters**: 1,516 (calculated below)
- **Connections**: 10 × 10 × 1,516 = **151,600**

### The Partial Connection Table (VERY IMPORTANT)

This is one of the most clever design decisions in the paper. In C3, **NOT every feature map is connected to all 6 S2 feature maps**. Instead, each C3 feature map is connected to a specific **subset** of S2 feature maps, according to this table:

```
                    C3 Feature Maps
              0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
S2 map 0:     ■  .  .  .  ■  ■  ■  .  .  ■  ■  ■  ■  .  ■  ■
S2 map 1:     ■  ■  .  .  .  ■  ■  ■  .  .  ■  ■  ■  ■  .  ■
S2 map 2:     ■  ■  ■  .  .  .  ■  ■  ■  .  .  ■  .  ■  ■  ■
S2 map 3:     .  ■  ■  ■  .  .  ■  ■  ■  ■  .  .  ■  .  ■  ■
S2 map 4:     .  .  ■  ■  ■  .  .  ■  ■  ■  ■  .  ■  ■  .  ■
S2 map 5:     .  .  .  ■  ■  ■  .  .  ■  ■  ■  ■  .  ■  ■  ■

■ = connected, . = not connected
```

Let me explain what this table means:
- **C3 feature map 0** takes input from S2 maps {0, 1, 2} (3 inputs)
- **C3 feature map 1** takes input from S2 maps {1, 2, 3} (3 inputs)
- **C3 feature map 6** takes input from S2 maps {0, 1, 2, 3} (4 inputs)
- **C3 feature map 15** takes input from S2 maps {0, 1, 2, 3, 4, 5} (ALL 6 inputs)

### Counting Parameters for C3

- Feature maps 0–5: each connected to 3 S2 maps → 3×(5×5) + 1 = 76 params each → 6 × 76 = 456
- Feature maps 6–11: each connected to 4 S2 maps → 4×(5×5) + 1 = 101 params each → 6 × 101 = 606
- Feature maps 12–14: each connected to 4 S2 maps → 4×(5×5) + 1 = 101 params each → 3 × 101 = 303
- Feature map 15: connected to all 6 S2 maps → 6×(5×5) + 1 = 151 params

**Total: 456 + 606 + 303 + 151 = 1,516 trainable parameters**

### Why Partial Connections?

The paper gives two reasons:

**Reason 1: Keep the number of parameters manageable.** Full connections would mean 16 × 6 × (5×5) + 16 = 2,416 parameters. Partial connections use only 1,516.

**Reason 2: Force different feature maps to learn different things.** If all feature maps saw all inputs, they might all learn similar features. By giving each feature map a different subset of inputs, you **force** them to extract different, complementary features.

**Analogy**: Imagine 16 students studying for an exam, but each student only has access to some of the textbook chapters:
- Student A reads chapters 1-3 → learns about topic A
- Student B reads chapters 2-4 → learns about topic B
- Student C reads chapters 1-6 (all) → gets the big picture

Together, they cover more ground than if they all read the same chapters.

### How C3 Convolution Works With Multiple Input Maps

When a C3 feature map is connected to, say, 3 S2 maps, the convolution works like this:

```
C3 Feature Map 0 connected to S2 maps {0, 1, 2}:

  S2 map 0 (14×14)      S2 map 1 (14×14)      S2 map 2 (14×14)
        ↓                      ↓                      ↓
   Kernel 0-0 (5×5)     Kernel 0-1 (5×5)      Kernel 0-2 (5×5)
        ↓                      ↓                      ↓
   Result 0 (10×10)     Result 1 (10×10)      Result 2 (10×10)
        ↓                      ↓                      ↓
        └──────────── SUM ─────────────────────┘
                        ↓
                  + bias (1 value)
                        ↓
               Activation function
                        ↓
              C3 Feature Map 0 (10×10)
```

Each connection has its **own kernel** (own set of 25 weights). The results are summed pixel-by-pixel, a bias is added, and the activation function is applied.

---

## Layer S4: Second Sub-Sampling Layer

### Specifications
- **Input**: 10×10×16
- **Pooling region**: 2×2
- **Stride**: 2
- **Output size**: 5×5×16
- **Trainable parameters**: 16 × 2 = **32**
- **Connections**: 16 × 5 × 5 × 5 = **2,000**

Works exactly like S2, but on the 16 feature maps from C3.

After S4, each feature map is only 5×5 = 25 pixels. These 25 pixels represent high-level information about what's in the image — far from raw pixel values.

---

## Layer C5: Third Convolutional Layer (Fully Connected Convolution)

### Specifications
- **Input**: 5×5×16
- **Kernel size**: 5×5
- **Number of feature maps**: 120
- **Output size**: 1×1×120 (i.e., a vector of 120 values)
- **Trainable parameters**: 120 × (16×5×5 + 1) = 120 × 401 = **48,120**
- **Connections**: **48,120**

### Why Is This "Fully Connected"?

Because the kernel size (5×5) equals the input size (5×5), each kernel covers the **entire** input feature map. There's only one position to place it. So the output for each feature map is a single number.

```
S4 feature map (5×5)       C5 Kernel (5×5)         Output
┌─────────────┐            ┌─────────────┐
│ x x x x x  │            │ k k k k k  │
│ x x x x x  │     ×      │ k k k k k  │    =    single value
│ x x x x x  │            │ k k k k k  │
│ x x x x x  │            │ k k k k k  │
│ x x x x x  │            │ k k k k k  │
└─────────────┘            └─────────────┘
```

Each C5 unit is connected to ALL 16 S4 feature maps (unlike C3 which had partial connections). So each C5 unit has 16 × 25 = 400 weights + 1 bias = 401 parameters.

With 120 such units: 120 × 401 = 48,120 parameters.

This is essentially a fully connected layer, but the paper keeps calling it "convolutional" because if the input were larger than 5×5, it would slide like a regular convolution.

---

## Layer F6: Fully Connected Layer

### Specifications
- **Input**: 120 neurons (from C5)
- **Output**: 84 neurons
- **Trainable parameters**: 84 × (120 + 1) = **10,164**
- **Connections**: **10,164**
- **Activation**: Same scaled tanh: f(x) = 1.7159 × tanh(2/3 × x)

### Why 84?

This is a very specific number! The paper chose 84 because the output layer uses a special representation where each digit is encoded as a **7×12 bitmap** (7 × 12 = 84). More on this in the output layer section.

```
F6 is simply:

C5 output (120 values) → Linear transformation → 84 values → tanh activation

Mathematically:
F6_output = tanh(W × C5_output + b)

where W is 84×120 and b is 84×1
```

---

## Output Layer: Euclidean Radial Basis Function (RBF) Units

### Specifications
- **Input**: 84 neurons (from F6)
- **Output**: 10 units (one per digit 0-9)
- **Trainable parameters**: **0** (the target bitmaps are FIXED, not learned)

### How It Works (This Is Unique!)

Unlike modern networks that use softmax, LeNet-5 uses a very different output scheme:

Each output unit computes the **Euclidean distance** between the F6 output and a **fixed target pattern**:

```
y_i = Σ(j=0 to 83) (x_j - w_ij)²
```

Where:
- `x_j` = the j-th output of F6
- `w_ij` = the j-th component of the target pattern for class i
- `y_i` = the output for class i (the "distance" from the target)

**Smaller y_i = better match!** (This is opposite to modern networks where bigger = better)

### The 7×12 Bitmap Encoding

Each digit 0-9 is represented by a **stylized 7×12 bitmap** — like a tiny image of the digit drawn on a 7-column × 12-row grid:

```
Target for digit "0":        Target for digit "1":
  ┌─────────────┐              ┌─────────────┐
  │ . . ■ ■ ■ . .│              │ . . . ■ . . .│
  │ . ■ . . . ■ .│              │ . . ■ ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ ■ . . . . . ■│              │ . . . ■ . . .│
  │ . ■ . . . ■ .│              │ . . . ■ . . .│
  │ . . ■ ■ ■ . .│              │ . ■ ■ ■ ■ ■ .│
  │ . . . . . . .│              │ . . . . . . .│
  └─────────────┘              └─────────────┘

■ = +1, . = -1

Each bitmap is flattened to a vector of 84 values (+1 or -1).
```

### Why This Strange Encoding?

The paper gives a clever reason: this encoding is designed so that **confusable characters** have very different target patterns. For example:
- The digit "0" and the letter "O" look similar visually
- But their 7×12 bitmaps can be designed to be very different

This was designed with an eye toward extending the system beyond just digits to include all ASCII characters.

### Classification Decision

Given an input image, the network produces 10 distance values:
```
y_0 = 15.2  (distance to target pattern for "0")
y_1 = 18.7  (distance to target pattern for "1")
y_2 = 3.1   (distance to target pattern for "2")  ← SMALLEST
y_3 = 22.4  (distance to target pattern for "3")
...
y_9 = 19.8  (distance to target pattern for "9")
```

The predicted class = the one with the **smallest distance** → in this case, "2".

---

## Complete Architecture Summary

```
Layer    Type          Maps   Size    Kernel  Stride  Params    Connections
─────    ────          ────   ────    ──────  ──────  ──────    ───────────
Input    -             1      32×32   -       -       0         0
C1       Convolution   6      28×28   5×5     1       156       122,304
S2       Subsampling   6      14×14   2×2     2       12        5,880
C3       Convolution   16     10×10   5×5     1       1,516     151,600
S4       Subsampling   16     5×5     2×2     2       32        2,000
C5       Convolution   120    1×1     5×5     1       48,120    48,120
F6       Full connect  -      84      -       -       10,164    10,164
Output   RBF           -      10      -       -       0         840
─────────────────────────────────────────────────────────────────────────
TOTAL                                                 60,000    340,908
```

> [!IMPORTANT]
> **Key Insight**: The network has **340,908 connections** but only **~60,000 trainable parameters** because of weight sharing! This is a 5.7× reduction. Without weight sharing (in a fully connected network of similar size), you'd need millions of parameters, and the network would overfit catastrophically.

---

## Complete Data Flow Example

Let's trace what happens when you feed an image of "7" through LeNet-5:

```
Step 1: INPUT
  32×32 image of "7" (1,024 pixels)
  
  . . . . . . . . . . . . . .
  . . . . . . . . . . . . . .
  . . ■ ■ ■ ■ ■ ■ ■ . . . . .     "7" with a horizontal bar on top
  . . . . . . . . ■ . . . . .      and a diagonal stroke going down
  . . . . . . . ■ . . . . . .
  . . . . . . ■ . . . . . . .
  . . . . . ■ . . . . . . . .
  . . . . ■ . . . . . . . . .
  . . . . ■ . . . . . . . . .
  . . . . . . . . . . . . . .
  . . . . . . . . . . . . . .

Step 2: C1 (6 feature maps, 28×28 each)
  Feature map 1 might show strong activation where horizontal edges were detected:
  ┌────────────────────┐
  │ . . ■ ■ ■ ■ ■ ■ . │  ← detects the top bar of "7"
  │ . . . . . . . . . │
  │ . . . . . . . . . │
  └────────────────────┘
  
  Feature map 2 might show strong activation where diagonal edges were detected:
  ┌────────────────────┐
  │ . . . . . . . . . │
  │ . . . . . . . ■ . │  ← detects the diagonal stroke
  │ . . . . . . ■ . . │
  │ . . . . . ■ . . . │
  └────────────────────┘

Step 3: S2 (6 feature maps, 14×14 each)
  Each feature map is halved in each dimension.
  The representation becomes more robust to slight shifts.

Step 4: C3 (16 feature maps, 10×10 each)
  Now the network combines low-level features:
  Feature map X might respond to "horizontal bar + diagonal" = characteristic of "7"
  Feature map Y might respond to "loop" = characteristic of "0" or "8"

Step 5: S4 (16 feature maps, 5×5 each)
  Further spatial reduction. Each 5×5 map is a highly compressed
  representation of what the network "sees."

Step 6: C5 (120 values)
  Each value captures a different high-level combination of features.
  This is like a "feature fingerprint" of the input.

Step 7: F6 (84 values)
  The 120-dim vector is transformed into an 84-dim vector
  that should be close to the target bitmap pattern for "7".

Step 8: OUTPUT (10 distances)
  Distance to "0" pattern: 45.2   (far → not "0")
  Distance to "1" pattern: 38.1   (far → not "1")
  Distance to "2" pattern: 41.7   (far → not "2")
  Distance to "3" pattern: 39.3   (far → not "3")
  Distance to "4" pattern: 44.8   (far → not "4")
  Distance to "5" pattern: 42.1   (far → not "5")
  Distance to "6" pattern: 40.5   (far → not "6")
  Distance to "7" pattern:  2.3   (CLOSE → it's "7"!) ✓
  Distance to "8" pattern: 43.9   (far → not "8")
  Distance to "9" pattern: 37.4   (far → not "9")
  
  Answer: "7" (minimum distance)
```
