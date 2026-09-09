# Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)

**Authors:** Karen Simonyan & Andrew Zisserman  
**Affiliation:** Visual Geometry Group (VGG), University of Oxford  
**Published:** ICLR 2015  
**arXiv:** [1409.1556](https://arxiv.org/abs/1409.1556)

---

## Table of Contents

1. [Paper at a Glance](#1-paper-at-a-glance)
2. [The Problem & Motivation](#2-the-problem--motivation)
3. [The Core Idea: Depth with Tiny Filters](#3-the-core-idea-depth-with-tiny-filters)
4. [Architecture in Detail](#4-architecture-in-detail)
   - 4.1 [Input Preprocessing](#41-input-preprocessing)
   - 4.2 [Convolutional Layers](#42-convolutional-layers)
   - 4.3 [Max-Pooling Layers](#43-max-pooling-layers)
   - 4.4 [Fully Connected Layers](#44-fully-connected-layers)
   - 4.5 [Activation Functions (ReLU)](#45-activation-functions-relu)
   - 4.6 [No Local Response Normalisation (LRN)](#46-no-local-response-normalisation-lrn)
5. [The 6 Configurations (A through E)](#5-the-6-configurations-a-through-e)
6. [Why 3×3 Filters? The Key Insight](#6-why-3×3-filters-the-key-insight)
   - 6.1 [Receptive Field Equivalence](#61-receptive-field-equivalence)
   - 6.2 [More Non-Linearity](#62-more-non-linearity)
   - 6.3 [Fewer Parameters (Regularisation Effect)](#63-fewer-parameters-regularisation-effect)
7. [The Role of 1×1 Convolutions](#7-the-role-of-1×1-convolutions)
8. [Training Methodology](#8-training-methodology)
   - 8.1 [Optimiser & Hyperparameters](#81-optimiser--hyperparameters)
   - 8.2 [Weight Initialisation Strategy](#82-weight-initialisation-strategy)
   - 8.3 [Data Augmentation](#83-data-augmentation)
   - 8.4 [Training Scale (S) — Single-Scale vs Multi-Scale](#84-training-scale-s--single-scale-vs-multi-scale)
9. [Testing Methodology](#9-testing-methodology)
   - 9.1 [Dense Evaluation (Fully-Convolutional Trick)](#91-dense-evaluation-fully-convolutional-trick)
   - 9.2 [Multi-Scale Testing](#92-multi-scale-testing)
   - 9.3 [Multi-Crop Evaluation](#93-multi-crop-evaluation)
10. [Experiments & Results](#10-experiments--results)
    - 10.1 [Single-Scale Results](#101-single-scale-results)
    - 10.2 [Multi-Scale Results](#102-multi-scale-results)
    - 10.3 [Dense vs Multi-Crop](#103-dense-vs-multi-crop)
    - 10.4 [Model Ensemble (Fusion)](#104-model-ensemble-fusion)
    - 10.5 [Comparison with State of the Art](#105-comparison-with-state-of-the-art)
11. [Localisation Task](#11-localisation-task)
12. [Transfer Learning & Generalisation](#12-transfer-learning--generalisation)
13. [Parameter Count Analysis](#13-parameter-count-analysis)
14. [VGG16 Layer-by-Layer Walkthrough](#14-vgg16-layer-by-layer-walkthrough)
15. [Historical Context & Legacy](#15-historical-context--legacy)
16. [Key Takeaways](#16-key-takeaways)

---

## 1. Paper at a Glance

| Aspect | Detail |
|---|---|
| **Main Question** | How does increasing the *depth* of a CNN affect image classification accuracy? |
| **Key Finding** | Pushing depth to 16–19 layers using only 3×3 filters significantly improves accuracy |
| **Dataset** | ImageNet ILSVRC-2012 (1.3M training images, 1000 classes) |
| **Best Result** | 7.3% top-5 test error (single model), 6.8% (ensemble of 2) |
| **ILSVRC 2014** | 🥇 1st place Localisation, 🥈 2nd place Classification |
| **Famous Models** | VGG16 (config D) and VGG19 (config E) |

> **One-sentence summary:** "Go deeper, but keep it simple — use only 3×3 convolution filters and stack many layers."

---

## 2. The Problem & Motivation

### Context: The ImageNet Challenge

By 2014, Convolutional Neural Networks (CNNs) had already proven their power. In 2012, AlexNet shattered previous records on the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC), achieving a top-5 error rate of ~16.4%. This was a massive leap from traditional hand-crafted features.

But a fundamental question remained:

> **"What matters more for accuracy — the specific architecture tricks, or simply making the network deeper?"**

Previous improvements after AlexNet (e.g., ZFNet in 2013) focused on things like:
- Using smaller filter sizes in the first layer (7×7 instead of 11×11)
- Using smaller strides
- Testing at multiple scales

The VGGNet authors took a different approach. They asked:

> "What if we fix everything else and *only* increase the depth?"

### Example: The Analogy

Think of building a house:
- **AlexNet's approach:** Use a few very large bricks (11×11, 7×7 filters) to build a short, wide building.
- **VGGNet's approach:** Use many tiny, uniform bricks (3×3 filters) to build a tall, elegant skyscraper.

The VGGNet authors showed that the skyscraper approach wins — and wins decisively.

---

## 3. The Core Idea: Depth with Tiny Filters

The central thesis of VGGNet can be distilled into three principles:

1. **Use only 3×3 convolution filters** — the smallest possible filter that can still capture spatial information (left/right, up/down, center).
2. **Stack many of them** — go deep (16–19 weight layers).
3. **Keep the design uniform and simple** — follow a repeating pattern of conv blocks followed by pooling.

This is in stark contrast to prior work:

| Model | Year | First Conv Filter | Total Depth |
|---|---|---|---|
| AlexNet | 2012 | 11×11 stride 4 | 8 layers |
| ZFNet | 2013 | 7×7 stride 2 | 8 layers |
| **VGGNet** | **2014** | **3×3 stride 1** | **16–19 layers** |

---

## 4. Architecture in Detail

### 4.1 Input Preprocessing

**Input:** A fixed-size **224×224 RGB** image.

**The only preprocessing applied:** Subtract the mean RGB value (computed over the entire training set) from each pixel.

#### Example
Suppose the training set has a mean pixel value of `[R=123.68, G=116.78, B=103.94]`. For any input pixel with values `[180, 200, 150]`, the preprocessed pixel becomes:

```
[180 - 123.68, 200 - 116.78, 150 - 103.94] = [56.32, 83.22, 46.06]
```

**Why subtract the mean?** This centres the data around zero, which helps gradient-based optimisation. Without this, all pixel values are positive (0–255), creating a bias that slows down learning.

> **Analogy:** Imagine trying to balance on a seesaw. If everyone is standing on one side (all positive values), it's hard to find balance. Mean subtraction moves people to both sides, making balance (convergence) easier.

---

### 4.2 Convolutional Layers

All convolutional layers use:

| Parameter | Value |
|---|---|
| Filter size | **3×3** |
| Stride | **1** pixel |
| Padding | **1** pixel (to preserve spatial resolution) |

#### How Padding Preserves Resolution — A Concrete Example

Consider a **5×5** input feature map convolved with a **3×3** filter:

- **Without padding:** The output is (5 - 3 + 1) = **3×3** → spatial resolution is lost.
- **With 1-pixel padding:** The input becomes effectively 7×7, and the output is (7 - 3 + 1) = **5×5** → resolution preserved!

The general formula:
```
Output size = (Input + 2×Padding - Filter) / Stride + 1
            = (5 + 2×1 - 3) / 1 + 1
            = 5
```

This is critical in VGGNet: spatial resolution is reduced *only* by max-pooling, never by convolution. This gives fine-grained control over when and how the spatial dimensions shrink.

#### Channel (Width) Progression

The number of filters (channels) follows a doubling pattern after each max-pooling:

```
64 → 128 → 256 → 512 → 512
```

**Example:** After the first two conv layers, you have 64 feature maps (each 224×224). After pooling and the next conv layers, you have 128 feature maps (each 112×112). The channels get richer as the spatial size shrinks — more "what" information, less "where" information.

> **Analogy:** Imagine summarising a book. At first you keep many details (high spatial resolution, few channels). As you summarise more, you lose specifics (lower resolution) but gain higher-level understanding (more channels). "Chapter about a dog in a park" → "animal scene" → "outdoor photograph."

---

### 4.3 Max-Pooling Layers

| Parameter | Value |
|---|---|
| Window size | **2×2** |
| Stride | **2** |

There are exactly **5 max-pooling layers** in every VGGNet configuration. Not every conv layer is followed by pooling — pooling only happens between "blocks" of conv layers.

#### Example: How Max-Pooling Works

Given a 4×4 feature map:
```
[1  3  2  4]
[5  6  7  8]
[9  2  1  0]
[3  4  5  6]
```

With a 2×2 window and stride 2, we take the maximum in each 2×2 block:
```
[max(1,3,5,6)  max(2,4,7,8)]   =   [6  8]
[max(9,2,3,4)  max(1,0,5,6)]       [9  6]
```

The 4×4 map becomes 2×2. Each pooling layer **halves** the spatial dimensions.

#### Spatial Dimension Progression Through VGGNet

```
224×224 → [conv block 1] → 224×224 → POOL → 112×112
         → [conv block 2] → 112×112 → POOL → 56×56
         → [conv block 3] → 56×56   → POOL → 28×28
         → [conv block 4] → 28×28   → POOL → 14×14
         → [conv block 5] → 14×14   → POOL → 7×7
```

---

### 4.4 Fully Connected Layers

After all conv + pooling layers, VGGNet uses **3 fully-connected (FC) layers:**

| Layer | Output Size | Purpose |
|---|---|---|
| FC-1 | 4096 | High-level feature combination |
| FC-2 | 4096 | Further feature abstraction |
| FC-3 | 1000 | One score per ImageNet class |
| Softmax | 1000 | Convert scores to probabilities |

#### Example: From Feature Map to FC Layer

After the last pooling, you have **512 feature maps of size 7×7**. These are *flattened* into a single vector:

```
512 × 7 × 7 = 25,088-dimensional vector
```

This vector is then fed into FC-1, which maps it to 4096 dimensions. Think of this as the network combining all the spatial features it learned into a holistic representation.

#### Softmax at the End

The final FC layer outputs 1000 raw scores (logits). Softmax converts these to probabilities:

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

**Example:** If the raw scores for three classes are `[2.0, 1.0, 0.1]`:
```
exp(2.0) = 7.39, exp(1.0) = 2.72, exp(0.1) = 1.11
Sum = 11.22
Probabilities = [0.659, 0.242, 0.099]
```

The network predicts class 1 with 65.9% confidence.

---

### 4.5 Activation Functions (ReLU)

Every hidden layer (conv and FC, except the final softmax) uses **ReLU (Rectified Linear Unit):**

```
ReLU(x) = max(0, x)
```

#### Example
```
Input:  [-2.5,  3.7, -0.1,  1.2,  0.0, -4.0]
Output: [ 0.0,  3.7,  0.0,  1.2,  0.0,  0.0]
```

**Why ReLU?**
1. **Fast to compute** — just a threshold comparison.
2. **No vanishing gradient for positive values** — the gradient is either 0 or 1, never a small fraction.
3. **Sparsity** — many neurons output 0, creating sparse (efficient) representations.

> **Analogy:** ReLU is like a bouncer at a club: if you're positive (above 0), you get in as you are. If you're negative, you're turned away (set to 0). No complicated decisions, just a simple threshold.

---

### 4.6 No Local Response Normalisation (LRN)

AlexNet used **Local Response Normalisation (LRN)**, a technique that normalises the activity of a neuron based on its neighbours (inspired by lateral inhibition in neuroscience).

VGGNet tested this and found:

> **LRN does NOT improve accuracy, but increases memory and computation time.**

They tested configuration A with LRN (called A-LRN) and found no benefit. So all deeper models (B through E) skip LRN entirely.

#### Example: Why LRN Doesn't Help Here

LRN was designed for networks with few layers where feature maps might have redundant, highly correlated activations. In deeper networks like VGGNet, the many stacked layers with ReLU already provide enough diversity and normalisation through depth itself.

> **Analogy:** LRN is like adding a spell-checker to a document that's already been proofread 16 times. At some point, the extra checking adds cost but no benefit.

---

## 5. The 6 Configurations (A through E)

The paper evaluates 6 architectures of increasing depth:

| Config | Depth (weight layers) | Conv Layers | FC Layers | Key Feature |
|---|---|---|---|---|
| **A** | 11 | 8 | 3 | Baseline |
| **A-LRN** | 11 | 8 | 3 | A + Local Response Normalisation |
| **B** | 13 | 10 | 3 | Added 2 more conv layers |
| **C** | 16 | 13 | 3 | Some 1×1 conv layers added |
| **D (VGG16)** | 16 | 13 | 3 | All conv layers are 3×3 |
| **E (VGG19)** | 19 | 16 | 3 | Even deeper — 16 conv layers |

### Detailed Architecture Table

```
Config A (11)      Config B (13)      Config C (16)      Config D (16)      Config E (19)
                                                          = VGG16            = VGG19
───────────────    ───────────────    ───────────────    ───────────────    ───────────────
Input: 224×224 RGB
───────────────    ───────────────    ───────────────    ───────────────    ───────────────
conv3-64           conv3-64           conv3-64           conv3-64           conv3-64
                   conv3-64           conv3-64           conv3-64           conv3-64
───────────── maxpool ─────────────────────────────────────────────────────────────────────
conv3-128          conv3-128          conv3-128          conv3-128          conv3-128
                   conv3-128          conv3-128          conv3-128          conv3-128
───────────── maxpool ─────────────────────────────────────────────────────────────────────
conv3-256          conv3-256          conv3-256          conv3-256          conv3-256
conv3-256          conv3-256          conv3-256          conv3-256          conv3-256
                                     conv1-256          conv3-256          conv3-256
                                                                           conv3-256
───────────── maxpool ─────────────────────────────────────────────────────────────────────
conv3-512          conv3-512          conv3-512          conv3-512          conv3-512
conv3-512          conv3-512          conv3-512          conv3-512          conv3-512
                                     conv1-512          conv3-512          conv3-512
                                                                           conv3-512
───────────── maxpool ─────────────────────────────────────────────────────────────────────
conv3-512          conv3-512          conv3-512          conv3-512          conv3-512
conv3-512          conv3-512          conv3-512          conv3-512          conv3-512
                                     conv1-512          conv3-512          conv3-512
                                                                           conv3-512
───────────── maxpool ─────────────────────────────────────────────────────────────────────
FC-4096
FC-4096
FC-1000
softmax
```

> **Notation:** `conv3-256` means a convolutional layer with 3×3 filters and 256 output channels.  
> `conv1-256` means a 1×1 convolution with 256 output channels.

### Parameter Counts

| Config | Depth | Parameters (millions) |
|---|---|---|
| A | 11 | 133 |
| B | 13 | 133 |
| C | 16 | 134 |
| D (VGG16) | 16 | 138 |
| E (VGG19) | 19 | 144 |

**Key observation:** Despite being much deeper, VGG19 has only ~144M parameters — comparable to shallower networks with larger filters (OverFeat had 144M parameters with fewer layers). The tiny 3×3 filters are what make this possible.

---

## 6. Why 3×3 Filters? The Key Insight

This is the most important contribution of the paper. Let's understand it deeply.

### 6.1 Receptive Field Equivalence

**Receptive field:** The region of the input image that a single neuron in a deeper layer "sees" (i.e., is influenced by).

#### The Math: Stacking 3×3 = Larger Receptive Fields

**Two 3×3 layers** stacked (no pooling between them) = effective receptive field of **5×5**:

```
Layer 1: Each neuron sees a 3×3 region of the input
Layer 2: Each neuron sees a 3×3 region of Layer 1's output
         But each of those Layer 1 neurons sees 3×3 of the input
         → Layer 2 neuron effectively sees 5×5 of the input

Calculation: (3 - 1) + (3 - 1) + 1 = 5
```

**Three 3×3 layers** stacked = effective receptive field of **7×7**:

```
(3 - 1) + (3 - 1) + (3 - 1) + 1 = 7
```

#### Visual Example

Imagine a 7×7 input grid. Here's what each approach "sees":

```
Single 7×7 filter:                    Three stacked 3×3 filters:

┌─────────────────┐                   Layer 3 output (1 neuron)
│ ● ● ● ● ● ● ● │                          ↑ sees 3×3 of layer 2
│ ● ● ● ● ● ● ● │                   Layer 2 (3×3 neurons)
│ ● ● ● ● ● ● ● │                          ↑ each sees 3×3 of layer 1
│ ● ● ● ● ● ● ● │   ≡               Layer 1 (5×5 neurons)
│ ● ● ● ● ● ● ● │                          ↑ each sees 3×3 of input
│ ● ● ● ● ● ● ● │                   Input (7×7)
│ ● ● ● ● ● ● ● │
└─────────────────┘                   Same 7×7 area is covered!
```

Both approaches cover the same 7×7 area of the input — but the stacked approach has major advantages.

---

### 6.2 More Non-Linearity

Each convolutional layer is followed by a ReLU activation. So:

- **One 7×7 layer → 1 ReLU** → one chance to introduce non-linearity
- **Three 3×3 layers → 3 ReLUs** → three chances to introduce non-linearity

More non-linearity = more complex decision boundaries = the network can learn more complex functions.

#### Example: Classifying "Cat vs Dog"

With 1 ReLU, the network can only make simple, piecewise-linear decisions:
```
Decision: "if whiskers > threshold → cat"
```

With 3 ReLUs, the network can compose multiple decisions:
```
Decision 1: "detect edge-like structures"
Decision 2: "combine edges into whisker/ear-like shapes"  
Decision 3: "combine shapes into cat-face vs dog-face pattern"
```

Each ReLU adds a "decision checkpoint" that makes the overall function more discriminative.

---

### 6.3 Fewer Parameters (Regularisation Effect)

This is where the math gets elegant. Assume both input and output have **C channels.**

**Three 3×3 conv layers:**
```
Parameters = 3 × (3² × C × C) = 3 × 9C² = 27C²
```

**One 7×7 conv layer:**
```
Parameters = 7² × C × C = 49C²
```

**Difference: 49C² vs 27C² → the 7×7 layer has 81% more parameters!**

#### Concrete Example with C = 512

```
Three 3×3 layers: 27 × 512² = 27 × 262,144 = 7,077,888 parameters
One 7×7 layer:    49 × 512² = 49 × 262,144 = 12,845,056 parameters

Savings: 12.8M - 7.1M = 5.7M fewer parameters (45% reduction)
```

Fewer parameters means:
1. **Less memory** needed
2. **Less computation** per forward/backward pass
3. **Implicit regularisation** — fewer parameters = less capacity to overfit
4. It's like **forcing a decomposition** of the 7×7 filter into three smaller filters with non-linearity injected between them

> **Analogy:** Imagine writing a 500-word essay (one 7×7 filter) vs. writing three 170-word paragraphs (three 3×3 filters). The total word count is less, but the paragraph structure forces you to organise your thoughts (add non-linearity), making the result clearer and more structured.

---

## 7. The Role of 1×1 Convolutions

Configuration C uses **1×1 convolution filters** in some layers. What do these do?

### What a 1×1 Convolution Does

A 1×1 convolution operates on each spatial location independently, mixing only across channels:

```
Input: H × W × C_in
Filter: 1 × 1 × C_in (one per output channel)
Output: H × W × C_out
```

It does NOT look at spatial neighbours. It's essentially a **per-pixel fully-connected layer** across channels.

### Why Use It?

1. **Adds non-linearity** (via the ReLU after it) without changing the receptive field.
2. **Changes channel dimensionality** — can increase or decrease the number of channels.
3. In VGGNet's case, C_in = C_out, so it's purely about adding non-linearity.

### Example: 1×1 Conv as Channel Mixing

Imagine a single pixel has activations across 256 channels: `[a₁, a₂, ..., a₂₅₆]`

A 1×1 conv with 256 filters computes 256 weighted sums:
```
output_channel_1 = w₁₁·a₁ + w₁₂·a₂ + ... + w₁,₂₅₆·a₂₅₆
output_channel_2 = w₂₁·a₁ + w₂₂·a₂ + ... + w₂,₂₅₆·a₂₅₆
...
```

Then ReLU is applied. This allows the network to learn new, non-linear combinations of features at each pixel location.

### The Finding: 3×3 is Still Better than 1×1

The paper found that Config C (with 1×1 convs) was better than Config B (fewer layers), confirming that extra non-linearity helps. But Config D (replacing those 1×1 convs with 3×3 convs) was even better.

> **Conclusion:** Non-linearity helps (C > B), but capturing spatial context is even more important (D > C). The 3×3 filter's ability to see its spatial neighbours gives it an edge over the purely channel-mixing 1×1 filter.

---

## 8. Training Methodology

### 8.1 Optimiser & Hyperparameters

| Hyperparameter | Value | Explanation |
|---|---|---|
| **Optimiser** | SGD + Momentum | Standard mini-batch gradient descent with momentum |
| **Batch size** | 256 | 256 images processed per weight update |
| **Momentum** | 0.9 | 90% of the previous gradient direction is retained |
| **Weight decay (L2)** | 5 × 10⁻⁴ | Penalises large weights to prevent overfitting |
| **Dropout** | 0.5 | Applied to FC-1 and FC-2 only |
| **Initial learning rate** | 10⁻² (0.01) | Starting point for weight updates |
| **LR schedule** | Divide by 10 when val accuracy plateaus | Decreased 3 times total |
| **Total iterations** | 370K (~74 epochs) | |

#### Momentum — Explained with Example

Without momentum, each weight update depends only on the current gradient:
```
w = w - lr × gradient
```

With momentum (0.9):
```
velocity = 0.9 × prev_velocity + gradient
w = w - lr × velocity
```

**Example:** Imagine pushing a ball down a hilly landscape.
- Without momentum: The ball stops at every small dip (local minimum).
- With momentum: The ball accumulates speed and rolls past small dips, finding deeper valleys (better minima).

#### Dropout — Explained with Example

During training, each neuron in FC-1 and FC-2 has a 50% chance of being "dropped" (output set to 0).

```
FC-1 output (no dropout): [2.3, 1.5, 0.8, 3.1, 2.0, 0.4]
Random mask:              [  1,   0,   1,   0,   1,   1 ]
FC-1 output (dropout):    [2.3, 0.0, 0.8, 0.0, 2.0, 0.4]
```

At test time, all neurons are active but outputs are scaled by 0.5 to compensate.

**Why?** Dropout forces the network to not rely on any single neuron. It's like studying for an exam by randomly removing some of your notes — you learn the material more robustly.

---

### 8.2 Weight Initialisation Strategy

Deep networks are notoriously hard to train because of the **vanishing/exploding gradient problem**. Bad initial weights can make gradients either shrink to zero or blow up to infinity.

**VGGNet's solution: Progressive training with pre-initialisation.**

#### Step 1: Train the shallowest network (Config A, 11 layers)
Config A is shallow enough to train from scratch with random initialisation:
- Weights sampled from a normal distribution: `N(0, 0.01)`
- Biases initialised to 0

#### Step 2: Use Config A's weights to initialise deeper networks
When training Config B, C, D, or E:
- Copy the **first 4 conv layers** and **last 3 FC layers** from the trained Config A
- Randomly initialise the new intermediate layers
- Allow all layers (including pre-initialised ones) to keep learning

#### Example: Building Knowledge Incrementally

Think of it like teaching:
1. First, teach a student basic shapes (train Config A).
2. Then, for a more advanced course (Config D), the student already knows shapes. You initialise them with that knowledge and teach more complex concepts on top.

> **Note from the paper:** After publication, the authors discovered that Xavier initialisation (Glorot & Bengio, 2010) works just as well, eliminating the need for this staged training. Modern frameworks (PyTorch, TensorFlow) use similar smart initialisations by default.

---

### 8.3 Data Augmentation

Three types of augmentation were used to artificially expand the training set:

#### 1. Random Cropping
From a rescaled image, randomly extract a 224×224 crop. Different crops are taken each iteration.

```
Original rescaled image (256×256):
┌──────────────────────────┐
│                          │
│     ┌──────────┐         │
│     │ 224×224  │         │  ← Random position each time
│     │  crop    │         │
│     └──────────┘         │
│                          │
└──────────────────────────┘
```

#### 2. Random Horizontal Flipping
Each crop has a 50% chance of being flipped left-to-right.

```
Original:    Flipped:
🐕 →          ← 🐕
```

This teaches the network that a dog facing left is the same as a dog facing right.

#### 3. Random RGB Colour Shift
Slight random perturbations to the colour channels (following the PCA-based method from AlexNet).

```
Original pixel [R=180, G=200, B=150]
After shift:   [R=183, G=197, B=152]  (small random change)
```

This makes the network robust to different lighting conditions.

---

### 8.4 Training Scale (S) — Single-Scale vs Multi-Scale

**S** = the smallest side of the rescaled training image (before cropping 224×224 from it).

#### Single-Scale Training (Fixed S)

Two fixed values were tested:
- **S = 256:** Classic setting. The crop covers most of the image.
- **S = 384:** Image is larger, so the crop captures a smaller portion (zoomed-in view).

```
S = 256:                           S = 384:
┌────────────────┐                ┌───────────────────────┐
│  ┌──────────┐  │                │                       │
│  │ 224×224  │  │                │    ┌──────────┐       │
│  │ (big     │  │                │    │ 224×224  │       │
│  │  portion)│  │                │    │ (small   │       │
│  └──────────┘  │                │    │  portion)│       │
└────────────────┘                │    └──────────┘       │
                                  └───────────────────────┘
```

For S=384, training was sped up by initialising with the S=256 weights and using a smaller learning rate (10⁻³).

#### Multi-Scale Training (Scale Jittering)

For each training image, **S is randomly sampled** from the range **[256, 512]**.

```
Image 1: S = 300 (medium zoom)
Image 2: S = 256 (zoom out)
Image 3: S = 480 (zoom in)
Image 4: S = 400 (zoom in more)
...
```

**Why is this powerful?** Objects in the real world appear at different scales. A cat close to the camera looks very different from a cat far away. By training on multiple scales, the network learns to recognise objects regardless of their size.

> **Analogy:** It's like studying faces. If you only ever see faces from 1 metre away, you might fail to recognise someone across a room. Multi-scale training is like practising recognition at 0.5m, 1m, 2m, 5m, etc.

The multi-scale models were initialised from the S=384 fixed-scale model.

---

## 9. Testing Methodology

### 9.1 Dense Evaluation (Fully-Convolutional Trick)

At test time, VGGNet uses a clever trick to avoid the inefficiency of cropping:

#### The Problem with Cropping
Traditional approach (AlexNet): extract multiple fixed-size crops (e.g., 10 crops), run each through the network, and average the predictions. But each crop requires a full forward pass — very wasteful.

#### The Solution: Convert FC Layers to Conv Layers

The key insight: a fully-connected layer is mathematically identical to a convolutional layer with a filter size equal to the input spatial size.

```
FC-1: Input 7×7×512, Output 4096
  ↓ equivalent to ↓
Conv: 7×7×512 filter, 4096 filters → output 1×1×4096
```

Similarly:
```
FC-2: Input 4096, Output 4096  →  Conv: 1×1×4096 filter
FC-3: Input 4096, Output 1000  →  Conv: 1×1×1000 filter
```

#### Why This Matters

Once the network is fully convolutional, you can feed in an image of **any size** (not just 224×224), and the output will be a **spatial map of class scores** instead of a single prediction.

```
Input: 224×224 → Output: 1×1×1000  (single prediction)
Input: 256×256 → Output: 2×2×1000  (4 predictions, one per spatial position)
Input: 384×384 → Output: 6×6×1000  (36 predictions)
```

These multiple predictions are then **averaged** to get the final class scores. This is equivalent to testing on many overlapping crops — but done in a **single forward pass!**

#### Example

```
Step 1: Rescale test image so smallest side = Q (e.g., 256)
Step 2: Run fully-convolutional network on the entire image
Step 3: Get a spatial map of scores (e.g., 2×2×1000)
Step 4: Average across spatial dimensions → 1000 class scores
Step 5: Also run on horizontally flipped image → another 1000 scores
Step 6: Average original + flipped scores → final prediction
```

---

### 9.2 Multi-Scale Testing

Just as multi-scale training helps, multi-scale testing also improves results.

For a **fixed-S model** (e.g., S=256), test at three scales:
```
Q = {S-32, S, S+32} = {224, 256, 288}
```

For a **multi-scale model** (S ∈ [256, 512]), test at:
```
Q = {S_min, 0.5(S_min + S_max), S_max} = {256, 384, 512}
```

The predictions at each scale are averaged for the final result.

---

### 9.3 Multi-Crop Evaluation

As an alternative to dense evaluation, the paper also tested **multi-crop evaluation** (like GoogLeNet):

- **50 crops per scale** (5×5 regular grid × 2 flips)
- **3 scales** → 150 total crops

**Finding:** Multi-crop performs slightly better than dense evaluation alone, and combining both (averaging their outputs) performs best of all. The authors hypothesise this is because dense and multi-crop evaluation handle convolution boundary conditions differently:

- **Dense:** Padding comes naturally from neighbouring pixels.
- **Multi-crop:** Padding is always zeros at the crop borders.

These different boundary treatments make them complementary.

---

## 10. Experiments & Results

### 10.1 Single-Scale Results

Testing each configuration at a single scale:

| Config | Depth | S (train) | Q (test) | Top-1 Error | Top-5 Error |
|---|---|---|---|---|---|
| A | 11 | 256 | 256 | 29.6% | 10.4% |
| A-LRN | 11 | 256 | 256 | 29.7% | 10.5% |
| B | 13 | 256 | 256 | 28.7% | 9.9% |
| C | 16 | 256 | 256 | 28.1% | 9.4% |
| D | 16 | 256 | 256 | 27.0% | 8.8% |
| D | 16 | 384 | 384 | 26.8% | 8.7% |
| D | 16 | [256,512] | 384 | 25.6% | 8.1% |
| E | 19 | 256 | 256 | 27.3% | 8.8% |
| E | 19 | 384 | 384 | 26.9% | 8.7% |
| E | 19 | [256,512] | 384 | 25.5% | 8.0% |

#### Key Observations

1. **LRN doesn't help:** A-LRN (10.5%) is worse than A (10.4%).
2. **Deeper is better:** 11 layers (10.4%) → 13 (9.9%) → 16 (8.8%) → 19 (8.8%).
3. **3×3 > 1×1:** Config D (8.8%, all 3×3) beats Config C (9.4%, some 1×1).
4. **Scale jittering is powerful:** D with jittering (8.1%) vs D with fixed S (8.8%).
5. **Depth saturates at 19 layers:** D and E perform similarly, but deeper might help on bigger datasets.

Also: a **shallow 5×5 net** (equivalent receptive field to Config B) had 7% higher top-1 error than B. This confirms: **deep + small filters > shallow + large filters.**

---

### 10.2 Multi-Scale Results

Testing at multiple scales (Q):

| Config | S (train) | Q (test) | Top-1 | Top-5 |
|---|---|---|---|---|
| D | 256 | {224, 256, 288} | 26.6% | 8.6% |
| D | 384 | {352, 384, 416} | 26.5% | 8.6% |
| D | [256,512] | {256, 384, 512} | 25.4% | 7.9% |
| E | 256 | {224, 256, 288} | 26.9% | 8.7% |
| E | 384 | {352, 384, 416} | 26.7% | 8.6% |
| E | [256,512] | {256, 384, 512} | **24.8%** | **7.5%** |

**Best single-model result: VGG19 with multi-scale training + multi-scale testing → 24.8% / 7.5%**

---

### 10.3 Dense vs Multi-Crop

Comparison for Config D (VGG16) with scale jittering:

| Evaluation Method | Top-1 Error | Top-5 Error |
|---|---|---|
| Dense only | 25.4% | 7.9% |
| Multi-crop only (150 crops) | 25.3% | 7.8% |
| Dense + Multi-crop (averaged) | **24.8%** | **7.4%** |

**Takeaway:** Both methods are good individually, but combining them gives the best results because they handle image boundaries differently.

---

### 10.4 Model Ensemble (Fusion)

Combining predictions from multiple models by averaging softmax outputs:

| Ensemble | Top-5 Test Error |
|---|---|
| 7 models (ILSVRC submission) | 7.3% |
| 2 best models (D + E, dense) | 7.0% |
| 2 best models (D + E, dense + multi-crop) | **6.8%** |

> **Key result:** Just 2 models combined achieved 6.8% — competitive with GoogLeNet's 6.7% which used a far more complex architecture.

---

### 10.5 Comparison with State of the Art

| Method | Year | Top-5 Test Error |
|---|---|---|
| AlexNet (Krizhevsky et al.) | 2012 | 16.4% |
| Clarifai (Zeiler & Fergus) | 2013 | 11.2% |
| OverFeat (Sermanet et al.) | 2013 | 13.0% |
| GoogLeNet (Szegedy et al.) | 2014 | **6.7%** |
| **VGGNet (this paper)** | **2014** | **6.8% (ensemble), 7.0% (single)** |

**VGGNet achieved the best single-model performance (7.0%), outperforming single GoogLeNet (7.9%) by 0.9%.** The ensemble was only 0.1% behind GoogLeNet's ensemble, despite being architecturally much simpler.

---

## 11. Localisation Task

Beyond classification, VGGNet also won 1st place in the ILSVRC-2014 **localisation** task with **25.3% error**.

### How Localisation Works

Instead of just saying "this is a dog," localisation requires drawing a **bounding box** around the object.

**Modification:** Replace the last FC layer (1000 classes) with a regression layer that predicts bounding box coordinates:

```
Classification output: [class_1_score, class_2_score, ..., class_1000_score]
Localisation output:   [center_x, center_y, width, height]
```

Two variants were tested:
- **Single-Class Regression (SCR):** One shared bounding box predictor for all classes (4 outputs)
- **Per-Class Regression (PCR):** One bounding box per class (4 × 1000 = 4000 outputs)

**Finding:** PCR worked better for VGGNet, unlike prior work (OverFeat) where SCR was better.

### Training Changes for Localisation
- **Loss function:** Euclidean loss (L2 distance between predicted and true bounding box) instead of cross-entropy
- **Initialisation:** Start from the pre-trained classification network
- Fine-tune **all layers** (not just FC layers — this was different from prior work and worked better)

### Localisation Results

| Method | Test Error |
|---|---|
| OverFeat (ILSVRC 2013 winner) | 29.9% |
| **VGGNet** | **25.3%** |

---

## 12. Transfer Learning & Generalisation

One of VGGNet's most impactful contributions was demonstrating that features learned on ImageNet **transfer beautifully** to other tasks.

### The Transfer Learning Pipeline

```
1. Take a VGGNet pre-trained on ImageNet
2. Remove the last FC layer (the 1000-class classifier)
3. Use the 4096-dimensional output of FC-2 as a feature vector
4. Train a simple linear SVM on the target dataset using these features
5. No fine-tuning of the CNN needed!
```

#### Example: Classifying Textures

Instead of training a whole new CNN on a small texture dataset (which would overfit), you:

1. Feed each texture image through pre-trained VGGNet
2. Extract the 4096-dim vector from FC-2
3. These vectors capture rich visual features (edges, textures, patterns)
4. A simple SVM can classify them with high accuracy

### Results on Other Datasets

| Dataset | Task | VGGNet (mAP/accuracy) | Previous Best |
|---|---|---|---|
| VOC-2007 | Image classification | 89.3% | 82.4% |
| VOC-2012 | Image classification | 89.0% | 83.2% |
| Caltech-101 | Image classification | 92.7% | 91.4% |
| Caltech-256 | Image classification | 86.2% | 77.6% |
| VOC-2012 | Action classification | 83.2% | 78.8% |

**The improvement on Caltech-256 is striking: 86.2% vs 77.6% — an 8.6% jump!**

### Why VGGNet Features Transfer So Well

The hierarchical features learned by VGGNet generalise because:

```
Layer 1-2:   Edges, colours           → universal to all images
Layer 3-5:   Textures, patterns       → broadly applicable
Layer 6-8:   Object parts             → transferable to related tasks
Layer 9-13:  Object-level features    → task-specific but still useful
FC layers:   High-level abstractions  → good general-purpose features
```

> **Analogy:** A person who has read thousands of books develops general comprehension skills (vocabulary, grammar, reasoning) that help them understand *any* new book — even on a completely different topic. VGGNet's ImageNet training builds similar "general visual comprehension."

---

## 13. Parameter Count Analysis

Where do all 138 million parameters (VGG16) actually live?

### Convolutional Layers — Surprisingly Few Parameters

Each conv layer has: `filter_size² × input_channels × output_channels + output_channels (bias)` parameters.

| Layer | Filter | Input Ch | Output Ch | Parameters |
|---|---|---|---|---|
| conv1_1 | 3×3 | 3 | 64 | 1,792 |
| conv1_2 | 3×3 | 64 | 64 | 36,928 |
| conv2_1 | 3×3 | 64 | 128 | 73,856 |
| conv2_2 | 3×3 | 128 | 128 | 147,584 |
| conv3_1 | 3×3 | 128 | 256 | 295,168 |
| conv3_2 | 3×3 | 256 | 256 | 590,080 |
| conv3_3 | 3×3 | 256 | 256 | 590,080 |
| conv4_1 | 3×3 | 256 | 512 | 1,180,160 |
| conv4_2 | 3×3 | 512 | 512 | 2,359,808 |
| conv4_3 | 3×3 | 512 | 512 | 2,359,808 |
| conv5_1 | 3×3 | 512 | 512 | 2,359,808 |
| conv5_2 | 3×3 | 512 | 512 | 2,359,808 |
| conv5_3 | 3×3 | 512 | 512 | 2,359,808 |
| **Total Conv** | | | | **~14.7M** |

### Fully Connected Layers — The Parameter Hogs

| Layer | Input | Output | Parameters |
|---|---|---|---|
| FC-1 | 7×7×512 = 25,088 | 4,096 | 102,764,544 |
| FC-2 | 4,096 | 4,096 | 16,781,312 |
| FC-3 | 4,096 | 1,000 | 4,097,000 |
| **Total FC** | | | **~123.6M** |

### The Shocking Ratio

```
Conv layers: ~14.7M parameters  → ~10.6% of total
FC layers:  ~123.6M parameters → ~89.4% of total
```

**Nearly 90% of VGGNet's parameters are in the fully-connected layers!** This is why later architectures (GoogLeNet, ResNet) replaced FC layers with global average pooling, dramatically reducing model size.

> **FC-1 alone** (102M parameters) contains **74%** of the entire network's parameters. This is because it connects every one of the 25,088 flattened features to each of the 4,096 output neurons.

---

## 14. VGG16 Layer-by-Layer Walkthrough

Let's trace a 224×224×3 image through VGG16 (Config D):

```
INPUT: 224 × 224 × 3                    (150,528 values)
│
├─ conv3-64 + ReLU    → 224 × 224 × 64
├─ conv3-64 + ReLU    → 224 × 224 × 64   (3.2M values)
├─ maxpool 2×2        → 112 × 112 × 64
│
├─ conv3-128 + ReLU   → 112 × 112 × 128
├─ conv3-128 + ReLU   → 112 × 112 × 128  (1.6M values)
├─ maxpool 2×2        → 56 × 56 × 128
│
├─ conv3-256 + ReLU   → 56 × 56 × 256
├─ conv3-256 + ReLU   → 56 × 56 × 256
├─ conv3-256 + ReLU   → 56 × 56 × 256    (803K values)
├─ maxpool 2×2        → 28 × 28 × 256
│
├─ conv3-512 + ReLU   → 28 × 28 × 512
├─ conv3-512 + ReLU   → 28 × 28 × 512
├─ conv3-512 + ReLU   → 28 × 28 × 512    (401K values)
├─ maxpool 2×2        → 14 × 14 × 512
│
├─ conv3-512 + ReLU   → 14 × 14 × 512
├─ conv3-512 + ReLU   → 14 × 14 × 512
├─ conv3-512 + ReLU   → 14 × 14 × 512    (100K values)
├─ maxpool 2×2        → 7 × 7 × 512
│
├─ FLATTEN             → 25,088
├─ FC-4096 + ReLU + Dropout(0.5)
├─ FC-4096 + ReLU + Dropout(0.5)
├─ FC-1000
├─ Softmax
│
OUTPUT: 1000 class probabilities
```

### What Each Block Learns (Intuitively)

```
Block 1 (64 filters):   Low-level features
                         → edges, corners, colour gradients
                         Example: "there's a vertical edge here"

Block 2 (128 filters):  Simple textures & patterns
                         → repeated edges, simple shapes
                         Example: "there's a striped pattern here"

Block 3 (256 filters):  Complex textures & object parts
                         → fur texture, brick pattern, eye shape
                         Example: "this looks like an eye-shaped region"

Block 4 (512 filters):  Object parts & assemblies
                         → face with eyes, wheel with tire
                         Example: "this region has a face-like arrangement"

Block 5 (512 filters):  High-level object features
                         → entire faces, car fronts, animal bodies
                         Example: "this is a golden retriever's face"

FC layers:              Combining everything for classification
                         → "This image is: 87% golden retriever,
                            5% Labrador, 3% Irish setter..."
```

---

## 15. Historical Context & Legacy

### Timeline of CNN Architectures

```
1998: LeNet-5         5 layers    → Handwritten digit recognition
2012: AlexNet         8 layers    → ImageNet breakthrough (16.4% error)
2013: ZFNet           8 layers    → Refined AlexNet (11.7% error)
2014: VGGNet          19 layers   → Depth matters! (6.8% error)
2014: GoogLeNet       22 layers   → Inception modules (6.7% error)
2015: ResNet          152 layers  → Skip connections (3.6% error)
```

### VGGNet's Lasting Contributions

1. **Proved depth matters:** Before VGGNet, it wasn't clear if going deeper would keep helping. VGGNet showed a clear monotonic improvement from 11 to 19 layers.

2. **Standardised 3×3 filters:** After VGGNet, 3×3 became the default convolution filter size in virtually all CNN architectures. Even ResNet, Inception, and modern architectures rely heavily on 3×3 convolutions.

3. **Transfer learning backbone:** For years after publication, VGG16 was the most popular feature extractor for transfer learning. It was used in:
   - Object detection (R-CNN, Fast R-CNN)
   - Semantic segmentation (FCN)
   - Neural style transfer
   - Image captioning
   - Super-resolution

4. **Simplicity as a design philosophy:** VGGNet showed that a clean, uniform architecture can compete with (and in single-model settings, outperform) more complex designs like GoogLeNet.

### Limitations

1. **Very large model:** 138M parameters and ~500MB on disk. Later models achieved better accuracy with far fewer parameters.
2. **Slow to train:** 2–3 weeks on 4 GPUs (NVIDIA Titan Black) in 2014.
3. **FC layers are wasteful:** 90% of parameters are in FC layers, which are less efficient than global average pooling (used in later architectures).
4. **No skip connections:** Depth beyond 19 layers was difficult due to vanishing gradients. ResNet solved this in 2015 with skip connections.

---

## 16. Key Takeaways

| # | Takeaway | Evidence |
|---|---|---|
| 1 | **Depth improves accuracy** | Error decreased monotonically from 11 to 19 layers |
| 2 | **Small 3×3 filters are better than large ones** | 3×3 stack beats equivalent 5×5 or 7×7 by a large margin |
| 3 | **More non-linearity (ReLU) helps** | Three ReLUs (3 layers) > one ReLU (1 layer) for same receptive field |
| 4 | **Fewer parameters = better generalisation** | 27C² params (three 3×3) vs 49C² (one 7×7) with better accuracy |
| 5 | **LRN is useless in deep networks** | A-LRN performed worse than A |
| 6 | **Scale jittering is a powerful augmentation** | Multi-scale training consistently outperformed fixed-scale |
| 7 | **Dense + multi-crop evaluation are complementary** | Combining both gave best results |
| 8 | **Simple ensembles help** | Just 2 models → 6.8% error (competitive with GoogLeNet's 6.7%) |
| 9 | **Deep features transfer well** | State-of-the-art on VOC, Caltech with simple SVM on top |
| 10 | **Simplicity can win** | Uniform 3×3 design competed with GoogLeNet's complex Inception modules |

---

> **Final Thought:** VGGNet's contribution was not a clever trick or a novel module — it was a *rigorous, controlled experiment* that answered a fundamental question: **"Does depth matter?"** The answer was a resounding yes, and this insight shaped every major architecture that followed.
