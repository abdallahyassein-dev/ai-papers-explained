# Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (R-CNN)

**Authors:** Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik  
**Institution:** UC Berkeley  
**Published:** CVPR 2014  
**Paper:** [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)

---

## Table of Contents

1. [The Problem: Why R-CNN Was Needed](#1-the-problem-why-r-cnn-was-needed)
2. [Big Picture: What R-CNN Does](#2-big-picture-what-r-cnn-does)
3. [The R-CNN Pipeline (Step by Step)](#3-the-r-cnn-pipeline-step-by-step)
4. [Step 1: Region Proposals with Selective Search](#4-step-1-region-proposals-with-selective-search)
5. [Step 2: Feature Extraction with CNN](#5-step-2-feature-extraction-with-cnn)
6. [Step 3: Classification with SVM](#6-step-3-classification-with-svm)
7. [Step 4: Bounding Box Regression](#7-step-4-bounding-box-regression)
8. [Training the R-CNN](#8-training-the-r-cnn)
9. [Transfer Learning & Domain Adaptation](#9-transfer-learning--domain-adaptation)
10. [IoU (Intersection over Union) Explained](#10-iou-intersection-over-union-explained)
11. [Non-Maximum Suppression (NMS)](#11-non-maximum-suppression-nms)
12. [Hard Negative Mining](#12-hard-negative-mining)
13. [Results and Impact](#13-results-and-impact)
14. [Strengths and Weaknesses](#14-strengths-and-weaknesses)
15. [Full Walkthrough Example](#15-full-walkthrough-example)
16. [Key Takeaways](#16-key-takeaways)

---

## 1. The Problem: Why R-CNN Was Needed

### Background: Object Detection Before R-CNN

Before R-CNN, the dominant approach for object detection on benchmarks like **PASCAL VOC** used **hand-crafted features** — most notably **HOG (Histogram of Oriented Gradients)** combined with a **Deformable Parts Model (DPM)**.

These methods had plateaued in performance. On PASCAL VOC 2012, the best systems achieved around **33–35% mAP** (mean Average Precision).

Meanwhile, in the **image classification** world, **CNNs** (specifically AlexNet) had just achieved a breakthrough at ImageNet 2012, dramatically outperforming hand-crafted features. The natural question was:

> **Can we bring the power of CNNs to object detection?**

### The Core Challenge

**Image classification** answers: *"What is in this image?"*  
**Object detection** answers: *"What objects are in this image AND where exactly are they?"*

The challenge is that CNNs were designed for classification (one label per image), but detection requires:
- Finding **multiple objects** in one image
- Drawing **bounding boxes** around each object
- Classifying each box independently

### Example: Classification vs. Detection

```
Image Classification:
┌─────────────────────┐
│                     │
│   🐕  🐈           │ → Label: "dog" (only one label)
│                     │
└─────────────────────┘

Object Detection:
┌─────────────────────┐
│  ┌────┐             │
│  │ 🐕 │  ┌────┐    │ → Box 1: "dog" (x=10, y=20, w=80, h=100)
│  └────┘  │ 🐈 │    │ → Box 2: "cat" (x=150, y=30, w=70, h=90)
│          └────┘     │
└─────────────────────┘
```

---

## 2. Big Picture: What R-CNN Does

R-CNN solves object detection by combining three ideas:

1. **Generate region proposals** — "Where might objects be?"
2. **Extract CNN features** — "What does each region look like?" (using a deep CNN)
3. **Classify each region** — "Is this region a dog? a cat? background?"

The name **R-CNN** stands for **Regions with CNN features**.

### The Architecture at a Glance

```
Input Image
     │
     ▼
┌──────────────────┐
│ Selective Search  │  → ~2000 region proposals (candidate bounding boxes)
└──────────────────┘
     │
     ▼ (for each region)
┌──────────────────┐
│ Warp to 227×227  │  → Resize each region to a fixed size
└──────────────────┘
     │
     ▼
┌──────────────────┐
│   CNN (AlexNet)   │  → Extract a 4096-dimensional feature vector
└──────────────────┘
     │
     ▼
┌──────────────────┐          ┌──────────────────────┐
│  SVM Classifiers  │          │ Bounding Box Regressor│
│ (one per class)   │          │ (refine box coords)   │
└──────────────────┘          └──────────────────────┘
     │                              │
     ▼                              ▼
  Class Scores              Refined Bounding Boxes
     │                              │
     └──────────┬───────────────────┘
                ▼
       Non-Maximum Suppression (NMS)
                │
                ▼
         Final Detections
```

---

## 3. The R-CNN Pipeline (Step by Step)

Let's walk through each stage in detail.

---

## 4. Step 1: Region Proposals with Selective Search

### What Is a Region Proposal?

Instead of using a brute-force **sliding window** approach (which would require evaluating millions of windows at different scales and aspect ratios), R-CNN uses a smarter method called **Selective Search** to generate around **~2000 candidate bounding boxes** per image.

### How Selective Search Works

Selective Search is a **bottom-up segmentation** method that groups pixels into regions based on:

- **Color similarity** — regions with similar colors merge together
- **Texture similarity** — regions with similar textures merge
- **Size** — smaller regions are prioritized for merging (to avoid one large region swallowing everything)
- **Fill** — how well regions fit into each other's bounding boxes

#### Step-by-Step Example

Imagine a photo of a park with a **red car**, a **green tree**, and a **blue sky**:

```
Step 1: Over-segmentation
┌──────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░ │  ← sky (blue pixels)
│ ░░░░░░░░░░░░░░░░░░░░ │
│ ████░░░░░▓▓▓▓▓▓░░░░░ │  ← car (red) + tree (green)
│ ████░░░░░▓▓▓▓▓▓░░░░░ │
│ ████░░░░░▓▓▓▓▓▓░░░░░ │
│ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  ← ground (brown pixels)
└──────────────────────┘

Step 2: Hierarchical grouping
- Blue pixels merge → sky region
- Red pixels merge → car region  
- Green pixels merge → tree region
- Brown pixels merge → ground region
- Then: car + ground merge → larger region
- Then: sky + tree merge → larger region
- Continue until you have the whole image as one region

Step 3: At each merge step, output the bounding box
→ Result: ~2000 bounding boxes at various scales
```

### Why Not Sliding Window?

| Approach | Number of Windows | Speed |
|----------|-------------------|-------|
| Sliding Window (all scales, aspect ratios) | ~100,000+ | Very slow |
| Selective Search | ~2,000 | Much faster |

Selective Search is **class-agnostic** — it doesn't know what objects are; it just finds "blobby" regions that *could* be objects.

---

## 5. Step 2: Feature Extraction with CNN

### The CNN Architecture

R-CNN uses **AlexNet** (the 2012 ImageNet winner) as its feature extractor. The architecture has 5 convolutional layers and 3 fully connected layers.

```
Input (227 × 227 × 3)
    │
    ▼
Conv1 (96 filters, 11×11, stride 4) → ReLU → Max Pool → Norm
    │
    ▼
Conv2 (256 filters, 5×5) → ReLU → Max Pool → Norm
    │
    ▼
Conv3 (384 filters, 3×3) → ReLU
    │
    ▼
Conv4 (384 filters, 3×3) → ReLU
    │
    ▼
Conv5 (256 filters, 3×3) → ReLU → Max Pool
    │
    ▼
FC6 (4096 neurons) → ReLU → Dropout
    │
    ▼
FC7 (4096 neurons) → ReLU → Dropout      ← This is the feature vector used!
    │
    ▼
FC8 (1000 neurons) → Softmax             ← This layer is replaced for detection
```

### The Warping Step

Each region proposal can be any size and any aspect ratio, but AlexNet requires exactly **227×227** pixels as input. So R-CNN **warps** (resizes) each proposal to 227×227, regardless of its original size.

#### Example of Warping

```
Original Region Proposal          After Warping
┌──────────────────┐              ┌───────────┐
│                  │              │           │
│                  │   resize     │           │
│   50 × 200 px   │  ────────►   │ 227 × 227 │
│                  │              │           │
│                  │              │           │
│                  │              └───────────┘
└──────────────────┘

(The tall thin region gets squished into a square)
```

> **Note:** The paper adds **16 pixels of padding** around each proposal before warping. This padding provides context from the surrounding image, which helps the CNN understand the object better.

### What Does the CNN Actually Learn?

The CNN learns a **hierarchy of features**:

```
Layer 1: Edges and color blobs
   ┌───┐  ┌───┐  ┌───┐
   │ / │  │ ─ │  │ █ │
   └───┘  └───┘  └───┘

Layer 2-3: Textures and parts
   ┌────┐  ┌────┐  ┌────┐
   │tire│  │eye │  │fur │
   └────┘  └────┘  └────┘

Layer 4-5: Object parts and shapes
   ┌─────┐  ┌─────┐  ┌─────┐
   │wheel│  │face │  │leg  │
   └─────┘  └─────┘  └─────┘

FC6-FC7: High-level semantic features
   → "This looks like a car" (4096-dim vector)
```

This is why the paper is called **"Rich Feature Hierarchies"** — the CNN builds increasingly complex features at each layer.

### The Feature Vector

For each of the ~2000 proposals, we get a **4096-dimensional feature vector** from layer `fc7`. This vector is a compact numerical representation of what the region "looks like."

```
Region Proposal #1  →  CNN  →  [0.23, -1.5, 0.87, ..., 0.12]  (4096 numbers)
Region Proposal #2  →  CNN  →  [1.10, 0.33, -0.45, ..., 0.78]  (4096 numbers)
...
Region Proposal #2000 → CNN → [-0.67, 0.91, 0.14, ..., -0.55] (4096 numbers)
```

---

## 6. Step 3: Classification with SVM

### Why SVM Instead of Softmax?

You might wonder: why not just use the CNN's softmax layer for classification? The paper actually tested this and found that **linear SVMs trained on CNN features outperformed the softmax** by about 4% mAP. The reasons are:

1. **Softmax was trained during fine-tuning** with a relatively loose definition of "positive" examples (IoU ≥ 0.5)
2. **SVMs can use stricter definitions** of positive/negative, which leads to better precision
3. SVMs are trained on the extracted features in a separate step, allowing more careful calibration

### How the SVMs Work

R-CNN trains **one binary SVM per class**. For PASCAL VOC (20 classes), there are 20 SVMs.

```
Feature vector (4096-d)
    │
    ├──► SVM_dog      → score: 0.92  ← "likely a dog"
    ├──► SVM_cat      → score: 0.05  ← "probably not a cat"
    ├──► SVM_car      → score: 0.01  ← "definitely not a car"
    ├──► SVM_person   → score: 0.03  ← "not a person"
    └──► SVM_...      → score: ...
```

Each SVM outputs a **confidence score**. After scoring all ~2000 proposals, we keep only the high-scoring ones and apply **Non-Maximum Suppression** (covered later).

### Example: Classifying a Region

```
Region Proposal: A crop of the image showing a dog

1. Warp to 227×227
2. Pass through CNN → get feature vector [0.23, -1.5, ...]
3. Feed to each SVM:
   - SVM_aeroplane:  -2.3 (negative = not an aeroplane)
   - SVM_bicycle:    -1.8
   - SVM_bird:       -0.5
   - SVM_dog:        +3.7 (positive = likely a dog!)    ✓
   - SVM_person:     -1.2
   - ... (15 more classes)
```

---

## 7. Step 4: Bounding Box Regression

### Why Do We Need This?

The region proposals from Selective Search are **rough estimates**. The bounding box might be slightly off — too big, too small, or shifted. Bounding box regression **refines** the coordinates to better fit the actual object.

### How It Works

For each proposal, we learn a **linear transformation** that maps the proposed box `P` to a ground-truth box `G`:

A bounding box is represented as 4 values:
- `(Px, Py)` = center coordinates
- `(Pw, Ph)` = width and height

The regressor learns 4 transformation functions:
```
dx(P) = (Gx - Px) / Pw       → horizontal shift (as fraction of width)
dy(P) = (Gy - Py) / Ph       → vertical shift (as fraction of height)
dw(P) = log(Gw / Pw)         → width scaling (in log space)
dh(P) = log(Gh / Ph)         → height scaling (in log space)
```

### Example: Bounding Box Correction

```
Before regression (Selective Search proposal):
┌─────────────────┐
│  ┌──────────┐   │
│  │          │   │
│  │   🐕     │   │  ← Proposal box (slightly too far left, too tall)
│  │          │   │
│  │          │   │
│  └──────────┘   │
└─────────────────┘

After bounding box regression:
┌─────────────────┐
│    ┌────────┐   │
│    │        │   │
│    │  🐕    │   │  ← Refined box (better fit!)
│    │        │   │
│    └────────┘   │
└─────────────────┘
```

### Numerical Example

```
Proposal P:       center=(100, 150), width=80, height=120
Ground Truth G:   center=(110, 145), width=90, height=100

Learned transformations:
  dx = (110 - 100) / 80  = 0.125   → shift right by 12.5% of width
  dy = (145 - 150) / 120 = -0.042  → shift up by 4.2% of height
  dw = log(90 / 80)      = 0.118   → widen by ~12%
  dh = log(100 / 120)    = -0.182  → shrink height by ~18%

At test time, given a new proposal P' = (200, 300, 60, 80):
  Predicted Gx = 200 + 0.125 × 60  = 207.5
  Predicted Gy = 300 + (-0.042) × 80 = 296.6
  Predicted Gw = 60 × exp(0.118)    = 67.5
  Predicted Gh = 80 × exp(-0.182)   = 66.7
```

> **Key detail:** The regression is **class-specific** — there's a separate regressor for each object class. A bounding box regressor for "car" learns different corrections than one for "person" because these objects have different typical shapes.

---

## 8. Training the R-CNN

Training R-CNN is a **multi-stage** process:

### Stage 1: Pre-train CNN on ImageNet

The CNN (AlexNet) is first trained on the **ImageNet** dataset (1.2 million images, 1000 classes) for image classification. This gives the CNN general visual knowledge about edges, textures, parts, and objects.

### Stage 2: Fine-tune CNN on Detection Data

The pre-trained CNN is then **fine-tuned** on the detection dataset (e.g., PASCAL VOC).

**Changes made:**
- Replace the 1000-class ImageNet output layer with a **(N+1)-class** layer (N object classes + 1 background class)
- For PASCAL VOC: 20 classes + 1 background = 21 outputs

**Training data construction:**
- **Positive example:** A region proposal with **IoU ≥ 0.5** with a ground-truth box
- **Negative example:** A region proposal with **IoU < 0.5** with all ground-truth boxes

**Training details:**
- SGD with learning rate **0.001** (1/10 of initial ImageNet rate)
- Each mini-batch: **32 positive + 96 negative** proposals = 128 total
- The low learning rate avoids destroying the pre-trained features

### Stage 3: Train SVMs

After fine-tuning, we **freeze the CNN** and extract features for all training proposals. Then we train one linear SVM per class.

**For SVMs, the positive/negative definitions are STRICTER:**
- **Positive:** Only the ground-truth bounding boxes themselves
- **Negative:** Proposals with **IoU < 0.3** with all ground-truth boxes
- **Ignored:** Proposals with 0.3 ≤ IoU < 0.5 (too ambiguous)

```
IoU Scale for SVM Training:
0.0          0.3          0.5          1.0
 │───────────│────────────│────────────│
 │  NEGATIVE │  IGNORED   │            │
 │  (< 0.3)  │(0.3 - 0.5) │            │
 │           │            │            │
 └───────────┴────────────┴────────────┘
                                    ▲
                              POSITIVE
                         (ground truth only)
```

### Stage 4: Train Bounding Box Regressors

Using the same frozen CNN features, train a **ridge regression** model for each class to predict bounding box corrections.

Only proposals with **IoU ≥ 0.6** with a ground-truth box are used for training (because the linear regression approximation only works when the proposal is already close to the ground truth).

---

## 9. Transfer Learning & Domain Adaptation

### What Is Transfer Learning?

Transfer learning is the idea of **using knowledge learned from one task to improve performance on another task**. This is one of the most important contributions of R-CNN.

### Example: Transfer Learning Analogy

```
Imagine you're a chef who has trained for 10 years in Italian cuisine.
Now you want to learn Japanese cuisine.

Option A (from scratch): Start over, learn everything from zero → slow
Option B (transfer learning): Leverage your existing skills 
  (knife skills, flavor understanding, plating) and adapt them → fast!

R-CNN does Option B:
- Pre-train on ImageNet (1.2M images, 1000 classes) → learn general visual features
- Fine-tune on PASCAL VOC (5K images, 20 classes) → adapt to detection
```

### Why Transfer Learning Works Here

The key insight is that **CNN features are general-purpose** in the early layers:

```
Layer      What It Learns           Transferable?
─────────────────────────────────────────────────
Conv1      Edges, color gradients    Very general ✓✓✓
Conv2      Textures, corners         General ✓✓✓
Conv3      Patterns, parts           Somewhat general ✓✓
Conv4      Object parts              Task-dependent ✓
Conv5      Object-level features     Task-specific ✓
FC6-7      Semantic understanding    Needs adaptation
```

### Domain Adaptation Results

The paper showed dramatic improvements from transfer learning:

| Method | Features | mAP on VOC 2010 |
|--------|----------|-----------------|
| DPM v5 | HOG (hand-crafted) | 33.4% |
| CNN (ImageNet only, no fine-tuning) | fc7 features | 44.2% |
| CNN (fine-tuned on VOC) | fc7 features | 54.2% |

> The fine-tuning step alone gave a **+10% boost** — proving that domain adaptation is critical.

---

## 10. IoU (Intersection over Union) Explained

IoU is a **metric that measures how much two bounding boxes overlap**. It's used everywhere in object detection for training labels, evaluation, and NMS.

### Formula

```
                  Area of Overlap
IoU = ────────────────────────────────────
           Area of Union

         Area(A ∩ B)
    = ─────────────────────
      Area(A) + Area(B) - Area(A ∩ B)
```

### Visual Example

```
Case 1: Perfect overlap (IoU = 1.0)
┌──────────┐
│ A and B  │
│ overlap  │
│ exactly  │
└──────────┘

Case 2: Partial overlap (IoU ≈ 0.5)
┌──────────┐
│    A     │
│      ┌───┼──────┐
│      │   │      │
└──────┼───┘      │
       │     B    │
       └──────────┘
  Overlap area / Union area ≈ 0.5

Case 3: No overlap (IoU = 0.0)
┌──────┐      ┌──────┐
│  A   │      │  B   │
└──────┘      └──────┘
```

### Numerical Example

```
Box A: top-left (10, 10), bottom-right (50, 50)  → Area = 40 × 40 = 1600
Box B: top-left (30, 30), bottom-right (70, 70)  → Area = 40 × 40 = 1600

Overlap: top-left (30, 30), bottom-right (50, 50) → Area = 20 × 20 = 400

Union = 1600 + 1600 - 400 = 2800

IoU = 400 / 2800 = 0.143
```

### IoU Thresholds in R-CNN

| Purpose | IoU Threshold | Meaning |
|---------|---------------|---------|
| CNN Fine-tuning Positive | ≥ 0.5 | "Close enough to be the object" |
| SVM Positive | Ground truth only | "Exact match" |
| SVM Negative | < 0.3 | "Clearly not the object" |
| BBox Regression Training | ≥ 0.6 | "Close enough for linear correction" |
| NMS | ≥ 0.3 | "Too much overlap — suppress one" |

---

## 11. Non-Maximum Suppression (NMS)

### The Problem: Duplicate Detections

After scoring all ~2000 proposals, multiple proposals often cover the **same object**. We need to keep only the best one.

### How NMS Works

```
Algorithm: Non-Maximum Suppression
─────────────────────────────────────
1. Sort all detections by confidence score (highest first)
2. Pick the highest-scoring detection → keep it
3. Remove all other detections that have IoU ≥ 0.3 with the kept detection
4. Repeat from step 2 with the remaining detections
5. Stop when no detections remain
```

### Example

```
Initial detections for class "dog":
┌───────────────────────────────┐
│                               │
│  ┌─────────┐                  │
│  │ Box A   │                  │
│  │ score:  │                  │
│  │ 0.95    ├──────┐           │
│  │   🐕    │Box B │           │
│  └─────────┤score:│           │
│            │0.88  │           │
│            └──────┘           │
│     ┌──────┐                  │
│     │Box C │                  │
│     │score:│                  │
│     │0.30  │                  │
│     └──────┘                  │
└───────────────────────────────┘

Step 1: Sort by score → [A(0.95), B(0.88), C(0.30)]

Step 2: Keep A (0.95)
  - IoU(A, B) = 0.65 > 0.3 → SUPPRESS B (too much overlap with A)
  - IoU(A, C) = 0.0  < 0.3 → KEEP C (different object)

Step 3: Keep C (0.30)
  - No more boxes

Result: Box A (dog, 0.95) and Box C (dog, 0.30)
→ Two dogs detected!
```

NMS is applied **independently for each class**. So a "dog" detection and a "cat" detection can overlap without suppressing each other.

---

## 12. Hard Negative Mining

### What Are Hard Negatives?

In object detection, the vast majority of proposals are **background** (negative). Most negatives are "easy" — they look nothing like any object (e.g., a patch of sky). But some negatives are **hard** — they look similar to objects but aren't (e.g., a tree stump that looks like a person).

### Why Hard Negative Mining Matters

If you train an SVM on all negatives equally, it will be dominated by easy negatives and won't learn to distinguish hard cases.

```
Easy Negative:                Hard Negative:
┌──────────┐                  ┌──────────┐
│          │                  │    /\    │
│  Plain   │                  │   /  \   │
│   Sky    │                  │  Tree    │
│          │                  │  Stump   │  ← Looks like a person!
│          │                  │   ||    │
└──────────┘                  └──────────┘
SVM easily says "not person"  SVM struggles — needs to see
                              more examples like this
```

### How R-CNN Does Hard Negative Mining

R-CNN uses a **standard hard negative mining approach**:

1. Initialize SVMs with a few positive examples and random negatives
2. Run the SVM on all training data
3. Collect **false positives** — negatives that the SVM incorrectly classified as positive (these are the "hard negatives")
4. Add these hard negatives to the training set
5. Re-train the SVM
6. Repeat until convergence

This iterative process ensures the SVM focuses on the most confusing examples.

---

## 13. Results and Impact

### Performance on PASCAL VOC

| Method | VOC 2007 mAP | VOC 2010 mAP | VOC 2012 mAP |
|--------|-------------|-------------|-------------|
| DPM v5 (previous best) | 33.7% | 33.4% | 30.4% |
| R-CNN (AlexNet) | 54.2% | 50.2% | 49.6% |
| R-CNN (VGG-16) | 66.0% | — | — |

> R-CNN achieved a **>20% absolute improvement** over the previous state-of-the-art. This was a massive leap.

### Performance on ILSVRC 2013 Detection

| Method | mAP |
|--------|-----|
| OverFeat | 24.3% |
| R-CNN (AlexNet) | 31.4% |
| R-CNN (Bounding Box Reg.) | 31.4% |

### Which CNN Layer Gives the Best Features?

The paper analyzed features from different CNN layers:

| Layer | VOC 2007 mAP (no fine-tuning) | VOC 2007 mAP (with fine-tuning) |
|-------|-------------------------------|--------------------------------|
| pool5 (6×6×256 = 9216-d) | 44.2% | 53.1% |
| fc6 (4096-d) | 43.0% | 54.1% |
| fc7 (4096-d) | 42.5% | 54.2% |

> **Key insight:** Without fine-tuning, the convolutional features (pool5) are best. After fine-tuning, the fully connected layers (fc6, fc7) improve dramatically — showing that fine-tuning primarily improves the higher layers.

---

## 14. Strengths and Weaknesses

### Strengths

| Strength | Explanation |
|----------|-------------|
| **Massive accuracy boost** | 20%+ improvement over hand-crafted features |
| **Transfer learning proof** | Showed CNN features transfer across tasks |
| **Modular design** | Easy to swap CNN backbone (AlexNet → VGG → etc.) |
| **Interpretable** | Each stage can be analyzed independently |
| **Scalable to more classes** | Just add more SVMs and regressors |

### Weaknesses

| Weakness | Explanation |
|----------|-------------|
| **Very slow at test time** | ~47 seconds per image (GPU) — runs CNN 2000 times! |
| **Multi-stage training** | CNN, SVMs, and regressors trained separately |
| **Disk space** | Feature caching requires hundreds of GB |
| **Warping distortion** | Squishing regions to 227×227 distorts objects |
| **Selective Search is fixed** | Region proposals can't be learned/improved |
| **Not end-to-end** | Can't backpropagate through the entire pipeline |

### Speed Breakdown

```
For one image at test time:
─────────────────────────────────────
Selective Search:    ~2 seconds
CNN forward pass:    ~13 seconds (×2000 regions = main bottleneck!)
SVM scoring:         ~10 seconds
NMS:                 ~1 second
Bounding Box Reg.:   ~1 second
─────────────────────────────────────
Total:              ~27–47 seconds per image
```

> This is why later methods (Fast R-CNN, Faster R-CNN) were developed — to address the speed problem.

---

## 15. Full Walkthrough Example

Let's trace through the **entire R-CNN pipeline** with a concrete example.

### Input: A photo of a street scene

```
┌──────────────────────────────────────┐
│                                      │
│     ☁️☁️☁️☁️☁️                       │
│                                      │
│   🏢🏢        🌳🌳                   │
│   🏢🏢        🌳🌳                   │
│                                      │
│        🚗         🧑                 │
│   ═══════════════════════════        │
└──────────────────────────────────────┘
```

### Step 1: Selective Search → ~2000 proposals

```
Some of the proposals:
┌──────┐  ┌────────┐  ┌──────────────┐  ┌────┐
│      │  │        │  │              │  │    │
│ Prop │  │ Prop   │  │  Prop #847   │  │#912│
│ #1   │  │ #423   │  │  (car area)  │  │    │
└──────┘  └────────┘  └──────────────┘  └────┘
(sky)     (building)    (car ✓)        (person ✓)
```

### Step 2: Warp and extract features for EACH proposal

```
Proposal #847 (car region)
  │
  ▼ Warp to 227×227
  │
  ▼ CNN Forward Pass
  │
  ▼ fc7 output: [0.82, -1.3, 2.1, ..., 0.45]  (4096 numbers)
```

### Step 3: Score with SVMs

```
Proposal #847 features → SVM scores:
  aeroplane:  -3.2
  bicycle:    -1.8
  bird:       -4.1
  boat:       -2.9
  bottle:     -3.5
  bus:         1.2
  car:         4.7  ← HIGHEST! This is likely a car
  cat:        -5.2
  ...
  person:     -2.1
  ...
```

### Step 4: Bounding Box Regression

```
Original proposal #847:  (x=120, y=280, w=100, h=60)
BBox regressor predicts:
  dx = +0.05  → shift right slightly
  dy = -0.02  → shift up slightly  
  dw = +0.08  → make slightly wider
  dh = +0.03  → make slightly taller

Refined box: (x=125, y=278.8, w=108.3, h=61.8)
```

### Step 5: NMS (for class "car")

```
After scoring all proposals for "car":
  Proposal #847: score 4.7, box (125, 279, 108, 62)  ← KEEP (highest)
  Proposal #851: score 3.9, box (122, 281, 105, 58)  ← SUPPRESS (IoU=0.78 with #847)
  Proposal #855: score 2.1, box (128, 282, 95, 55)   ← SUPPRESS (IoU=0.65 with #847)
```

### Final Output

```
Detection Results:
─────────────────────────────
Object: car     | Score: 4.7 | Box: (125, 279, 108, 62)
Object: person  | Score: 3.2 | Box: (310, 250, 45, 120)
```

---

## 16. Key Takeaways

### 1. R-CNN Proved That CNNs Dominate Object Detection
Before R-CNN, hand-crafted features (HOG, SIFT) were the norm. R-CNN showed a **20%+ mAP improvement** by using CNN features, effectively ending the era of hand-crafted features in detection.

### 2. Transfer Learning Is Powerful
Pre-training on ImageNet and fine-tuning on the target dataset was a game-changer. This pattern became the **default approach** for nearly all computer vision tasks.

### 3. The "Detect-then-Classify" Paradigm
R-CNN established the two-stage detection paradigm:
1. **Propose** candidate regions
2. **Classify** each region

This paradigm influenced all subsequent detectors (Fast R-CNN, Faster R-CNN, Mask R-CNN).

### 4. Feature Hierarchies Matter
The paper showed that deeper CNN layers learn increasingly abstract and task-specific features — and that fine-tuning is essential for adapting these features to new tasks.

### 5. Limitations Led to Future Innovations

```
R-CNN (2014)
  │ Problem: Too slow (CNN runs 2000 times)
  ▼
Fast R-CNN (2015)
  │ Solution: Share CNN computation across all proposals
  │ Problem: Selective Search is still slow
  ▼
Faster R-CNN (2015)
  │ Solution: Replace Selective Search with a Region Proposal Network (RPN)
  │ Problem: Can't do instance segmentation
  ▼
Mask R-CNN (2017)
  │ Solution: Add a segmentation branch
  ▼
Modern Detectors (YOLO, DETR, etc.)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **mAP** | Mean Average Precision — the main metric for object detection. Average of AP across all classes. |
| **AP** | Average Precision — area under the precision-recall curve for one class. |
| **IoU** | Intersection over Union — overlap measure between two boxes (0 to 1). |
| **NMS** | Non-Maximum Suppression — removes duplicate detections. |
| **Region Proposal** | A candidate bounding box that might contain an object. |
| **Selective Search** | An algorithm that generates ~2000 region proposals per image. |
| **Fine-tuning** | Taking a pre-trained model and training it further on a new dataset. |
| **Transfer Learning** | Using knowledge from one task (ImageNet) to help another (detection). |
| **SVM** | Support Vector Machine — a linear classifier used for scoring proposals. |
| **Bounding Box Regression** | A model that refines the coordinates of a proposed bounding box. |
| **Hard Negative Mining** | Iteratively finding and training on the most confusing negative examples. |
| **HOG** | Histogram of Oriented Gradients — a hand-crafted feature descriptor. |
| **DPM** | Deformable Parts Model — the dominant pre-CNN detection method. |
| **fc7** | The 7th (second fully connected) layer in AlexNet, producing a 4096-d vector. |
| **Warping** | Resizing a region proposal to a fixed size (227×227) for CNN input. |

---

## Appendix B: Mathematical Notation Summary

### Bounding Box Parameterization

A bounding box is described by its **center** and **dimensions**:

```
P = (Px, Py, Pw, Ph)   → proposed box
G = (Gx, Gy, Gw, Gh)   → ground-truth box
```

### Transformation Functions

```
tx(P) = (Gx - Px) / Pw
ty(P) = (Gy - Py) / Ph
tw(P) = log(Gw / Pw)
th(P) = log(Gh / Ph)
```

### Why Log for Width/Height?

Using `log` ensures that **scaling is symmetric**:
- Doubling the width: `log(2) ≈ 0.693`
- Halving the width: `log(0.5) ≈ -0.693`

Without log, doubling would be `+1.0` but halving would be `-0.5` — asymmetric!

### Ridge Regression for BBox

The bounding box regressor minimizes:

```
L = Σᵢ (tᵢ - wᵀφ(Pᵢ))² + λ||w||²

where:
  tᵢ     = target transformation
  w      = learned weights
  φ(Pᵢ)  = CNN features of proposal Pᵢ (4096-d from pool5)
  λ      = regularization parameter (λ = 1000 in the paper)
```

---

## Appendix C: Semantic Segmentation with R-CNN

The paper also briefly explores using R-CNN for **semantic segmentation** — labeling every pixel in an image with a class.

### Approach: Regions to Pixels

```
1. Run R-CNN normally → get class scores for each region
2. For each pixel, find all regions containing that pixel
3. Aggregate the scores → assign the pixel to the highest-scoring class
```

The authors tried three region-based strategies:
- **full:** Use the full region's CNN features
- **fg:** Use only the foreground (inside the region's segmentation mask)
- **full+fg:** Concatenate both → works best

### Results on VOC 2011 Segmentation

| Method | Mean IoU |
|--------|----------|
| O2P (previous best) | 46.4% |
| R-CNN (full+fg, fc7) | 47.9% |

While the improvement was modest, it proved that R-CNN features are **versatile enough** for both detection and segmentation.

---

> **Historical Note:** R-CNN was published in November 2013 (arXiv) and presented at CVPR 2014. It is one of the most cited papers in computer vision history and marks the beginning of the deep learning revolution in object detection.
