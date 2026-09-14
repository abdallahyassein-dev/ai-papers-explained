# 🧠 AI & Deep Learning Papers Explained (From Scratch)

[![Papers Covered](https://img.shields.io/badge/Roadmap%20Progress-12%2F49%20Completed-brightgreen.svg?style=flat-square)](#-milestone-papers-completed)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20العربية-blue.svg?style=flat-square)](#-overview--نظرة-عامة)
[![Code](https://img.shields.io/badge/Implementations-From%20Scratch%20(Python%20%26%20PyTorch)-orange.svg?style=flat-square)](#-from-scratch-implementations)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg?style=flat-square)](LICENSE)

> A curated, comprehensive repository of foundational and modern milestone AI & Deep Learning research papers — featuring **in-depth mathematical breakdowns**, **native from-scratch code implementations**, and **bilingual guides (Arabic & English)** following a progressive 49-paper learning roadmap.

---

## 📖 Overview | نظرة عامة

### English
This repository is designed as a deep-dive, practical companion for anyone looking to master the foundations and breakthrough milestones of Artificial Intelligence. Instead of high-level summaries, each paper here is accompanied by:
- **Comprehensive Breakdowns:** Step-by-step mathematical derivations, intuitive explanations of core mechanisms, and architectural diagrams.
- **From-Scratch Implementations:** Pure Python, NumPy, and native PyTorch implementations without relying on high-level abstraction libraries.
- **Original Research Papers:** Verified PDF copies preserved for offline study.

### العربية
هذا المستودع تم إنشاؤه ليكون مرجعاً شاملاً وعملياً لكل باحث ومتعلم في مجال الذكاء الاصطناعي وتعلم الآلة. لا يقتصر المستودع على إتاحة الأوراق البحثية، بل يقدم:
- **شروحات تفصيلية مبسطة وعميقة:** تفكيك كامل للرياضيات الكامنة وراء كل ابتكار، وشرح بنية النماذج باللغتين العربية والإنجليزية.
- **تطبيق عملي وبرمجة من الصفر (From Scratch):** بناء المعماريات باستخدام بايثون و PyTorch دون الاعتماد على مكتبات جاهزة لفهم كل معادلة عملياً.
- **خارطة طريق ممنهجة (49 ورقة بحثية):** مستوحاة من خارطة طريق أبحاث DeepMind تغطي أسس الذكاء الاصطناعي وحتى أحدث نماذج اللغة والرؤية التوليدية.

---

## 🏆 Milestone Papers Completed (12 / 49)

| # | Paper Title | Year | Category | In-Depth Guide / Summary | Code Implementation | Status |
| :-: | :--- | :-: | :---: | :---: | :---: | :---: |
| **01** | **Learning Representations by Back-propagating Errors**<br>*(Rumelhart, Hinton, Williams)* | 1986 | Foundations | [Breakdown](1%20-%20Learning%20representations%20by%20back-propagating%20errors/backpropagation_paper_breakdown.md) | [backpropagation.py](1%20-%20Learning%20representations%20by%20back-propagating%20errors/backpropagation.py) | ✅ Completed |
| **02** | **Gradient-Based Learning Applied to Document Recognition (LeNet-5)**<br>*(Yann LeCun et al.)* | 1998 | Computer Vision | [Guide Part 1](2%20-%20Gradient-Based%20Learning%20Applied%20to%20Document%20Recognition%20(LeNet-5)/lenet5_paper_explained_part1.md) • [Part 2](2%20-%20Gradient-Based%20Learning%20Applied%20to%20Document%20Recognition%20(LeNet-5)/lenet5_paper_explained_part2.md) | [lenet5-from-scratch/](2%20-%20Gradient-Based%20Learning%20Applied%20to%20Document%20Recognition%20(LeNet-5)/lenet5-from-scratch) | ✅ Completed |
| **03** | **A Neural Probabilistic Language Model (NPLM)**<br>*(Bengio et al.)* | 2003 | Language Models | [شرح بالعربي](3%20-%20A%20Neural%20Probabilistic%20Language%20Model/neural_language_model_explained.md) • [English Guide](3%20-%20A%20Neural%20Probabilistic%20Language%20Model/neural_language_model_explained_en.md) | [nplm_native.py](3%20-%20A%20Neural%20Probabilistic%20Language%20Model/nplm_native.py) • [demo.py](3%20-%20A%20Neural%20Probabilistic%20Language%20Model/demo.py) | ✅ Completed |
| **04** | **ImageNet: A Large-Scale Hierarchical Image Database**<br>*(Deng et al.)* | 2009 | Vision & Datasets | [Paper PDF](4%20-%20ImageNet%20-%20A%20Large-Scale%20Hierarchical%20Image%20Database/ImageNet_a_Large-Scale_Hierarchical_Image_Database.pdf) | — | ✅ Completed |
| **05** | **ImageNet Classification with Deep CNNs (AlexNet)**<br>*(Krizhevsky, Sutskever, Hinton)* | 2012 | Computer Vision | [AlexNet Guide](5%20-%20ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks%20(AlexNet)/README.md) | [demo_alexnet.py](5%20-%20ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks%20(AlexNet)/demo_alexnet.py) | ✅ Completed |
| **06** | **Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)**<br>*(Simonyan & Zisserman)* | 2014 | Computer Vision | [VGGNet Explained](6%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition%20(VGGNet)/VGGNet_Explained.md) | Architectural Notes | ✅ Completed |
| **07** | **Going Deeper with Convolutions (GoogLeNet / Inception v1)**<br>*(Szegedy et al.)* | 2014 | Computer Vision | [Inception Summary](7%20-%20Going%20Deeper%20with%20Convolutions%20(GoogLeNet)/going_deeper_with_convolutions_summary.md) | Architectural Notes | ✅ Completed |
| **08** | **Batch Normalization: Accelerating Training by Reducing Covariate Shift**<br>*(Ioffe & Szegedy)* | 2015 | Deep Learning Foundations | [Batch Norm Deep Dive](8%20-%20Batch%20Normalization/batch_normalization_2015.md) | Algorithmic Formulation | ✅ Completed |
| **09** | **Deep Residual Learning for Image Recognition (ResNet)**<br>*(He et al.)* | 2015 | Computer Vision | [ResNet Deep Dive](9%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition%20(ResNet)/resnet_explained.md) | Architectural Analysis | ✅ Completed |
| **10** | **Rich Feature Hierarchies for Accurate Object Detection (R-CNN)**<br>*(Girshick et al.)* | 2014 | Object Detection | [R-CNN Explained](10%20-%20Rich%20Feature%20Hierarchies%20(R-CNN)/RCNN_Explained.md) | Pipeline Breakdown | ✅ Completed |
| **11** | **Fast R-CNN**<br>*(Ross Girshick)* | 2015 | Object Detection | [Fast R-CNN Explained](11%20-%20Fast%20R-CNN/Fast%20R-CNN%20-%20Explained.md) • [PDF](11%20-%20Fast%20R-CNN/Fast%20R-CNN.pdf) | RoI Pooling & Multi-Task Loss Breakdown | ✅ Completed |
| **12** | **Faster R-CNN: Towards Real-Time Object Detection with RPN**<br>*(Ren, He, Girshick, Sun)* | 2015 | Object Detection | [Faster R-CNN Explained](12%20-%20Faster%20R-CNN/Faster_RCNN_Explained.md) • [PDF](12%20-%20Faster%20R-CNN/1506.01497v3.pdf) | RPN, Anchors & 4-Step Alternating Training | ✅ Completed |

---

## 🗺️ The Complete AI Research Roadmap (49 Papers)

The roadmap is categorized by research domain and based on [`DeepMind_AI_Paper_Roadmap.xlsx`](DeepMind_AI_Paper_Roadmap.xlsx):

### 1. Foundations & Early Vision
- [x] **1986** — *Learning representations by back-propagating errors* (Rumelhart et al.)
- [x] **1998** — *Gradient-Based Learning Applied to Document Recognition (LeNet-5)* (LeCun et al.)
- [x] **2003** — *A Neural Probabilistic Language Model (NPLM)* (Bengio et al.)
- [x] **2009** — *ImageNet: A Large-Scale Hierarchical Image Database* (Deng et al.)
- [ ] **2010** — *Understanding the difficulty of training deep feedforward neural networks (Xavier Initialization)* (Glorot & Bengio)

### 2. Deep CNN Revolution
- [x] **2012** — *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)* (Krizhevsky et al.)
- [x] **2014** — *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)* (Simonyan & Zisserman)
- [x] **2014** — *Going Deeper with Convolutions (GoogLeNet / Inception)* (Szegedy et al.)
- [x] **2015** — *Batch Normalization: Accelerating Deep Network Training* (Ioffe & Szegedy)
- [x] **2015** — *Deep Residual Learning for Image Recognition (ResNet)* (He et al.)
- [ ] **2015** — *Delving Deep into Rectifiers: Surpassing Human-Level Performance (He Initialization)* (He et al.)

### 3. Object Detection & Semantic Segmentation
- [x] **2014** — *Rich Feature Hierarchies for Accurate Object Detection (R-CNN)* (Girshick et al.)
- [x] **2015** — *Fast R-CNN* (Girshick)
- [x] **2015** — *Faster R-CNN: Towards Real-Time Object Detection with RPN* (Ren et al.)
- [ ] **2016** — *You Only Look Once: Unified, Real-Time Object Detection (YOLO)* (Redmon et al.)
- [ ] **2018** — *YOLOv3: An Incremental Improvement* (Redmon & Farhadi)
- [ ] **2015** — *Fully Convolutional Networks for Semantic Segmentation (FCN)* (Long et al.)
- [ ] **2015** — *U-Net: Convolutional Networks for Biomedical Image Segmentation* (Ronneberger et al.)
- [ ] **2017** — *Mask R-CNN* (He et al.)

### 4. Transformers & Foundation LLMs
- [ ] **2017** — *Attention Is All You Need* (Vaswani et al.)
- [ ] **2018** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al.)
- [ ] **2020** — *Language Models are Few-Shot Learners (GPT-3)* (Brown et al.)
- [ ] **2020** — *Scaling Laws for Neural Language Models* (Kaplan et al.)
- [ ] **2022** — *PaLM: Scaling Language Modeling with Pathways* (Chowdhery et al.)
- [ ] **2022** — *Training Compute-Optimal Large Language Models (Chinchilla)* (Hoffmann et al.)
- [ ] **2023** — *Gemini: A Family of Highly Capable Multimodal Models* (Gemini Team, Google)

### 5. Vision Transformers (ViT) & Multimodal AI
- [ ] **2020** — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)* (Dosovitskiy et al.)
- [ ] **2021** — *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al.)
- [ ] **2021** — *Training data-efficient image transformers & distillation through attention (DeiT)* (Touvron et al.)
- [ ] **2022** — *Flamingo: a Visual Language Model for Few-Shot Learning* (Alayrac et al.)

### 6. Self-Supervised Learning
- [ ] **2019** — *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)* (He et al.)
- [ ] **2020** — *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)* (Chen et al.)
- [ ] **2020** — *Bootstrap Your Own Latent (BYOL)* (Grill et al.)
- [ ] **2021** — *Masked Autoencoders Are Scalable Vision Learners (MAE)* (He et al.)

### 7. Generative AI & Diffusion Models
- [ ] **2013** — *Auto-Encoding Variational Bayes (VAE)* (Kingma & Welling)
- [ ] **2014** — *Generative Adversarial Networks (GANs)* (Goodfellow et al.)
- [ ] **2020** — *Denoising Diffusion Probabilistic Models (DDPM)* (Ho et al.)
- [ ] **2022** — *High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)* (Rombach et al.)

### 8. Reinforcement Learning & DeepMind Milestones
- [ ] **2013** — *Playing Atari with Deep Reinforcement Learning (DQN)* (Mnih et al.)
- [ ] **2015** — *Human-level control through deep reinforcement learning* (Mnih et al.)
- [ ] **2016** — *Mastering the game of Go with deep neural networks and tree search (AlphaGo)* (Silver et al.)
- [ ] **2017** — *Mastering Chess and Shogi by Self-Play with a General RL Algorithm (AlphaZero)* (Silver et al.)
- [ ] **2020** — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)* (Schrittwieser et al.)
- [ ] **2021** — *Highly accurate protein structure prediction with AlphaFold* (Jumper et al.)

### 9. Optimization, Regularization & Research Wisdom
- [ ] **2014** — *Adam: A Method for Stochastic Optimization* (Kingma & Ba)
- [ ] **2014** — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al.)
- [ ] **2015** — *A Survey on Deep Learning* (Schmidhuber)
- [ ] **2016** — *Deep Learning Textbook* (Goodfellow, Bengio, Courville)
- [ ] **2019** — *The Bitter Lesson* (Rich Sutton)

---

## 💻 From-Scratch Implementations

All code implementations are written with educational clarity in mind:
- [x] **Backpropagation from scratch**: Native matrix calculus, forward pass, gradient backward pass, and parameter updates.
- [x] **LeNet-5 from scratch**: Convolution, subsampling, fully connected layers, and MNIST digit classifier.
- [x] **NPLM from scratch**: Word embeddings, hidden projection layer, softmax output, and sentence generation demos.
- [x] **AlexNet Demo**: Complete convolutional pipeline, ReLU activations, Local Response Normalization, and Dropout.

---

## 🚀 How to Run the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdallahyassein-dev/ai-papers-explained.git
   cd ai-papers-explained
   ```

2. **Set up a Python environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torchvision numpy matplotlib
   ```

3. **Run any scratch implementation (e.g. Backpropagation):**
   ```bash
   python3 "1 - Learning representations by back-propagating errors/backpropagation.py"
   ```

4. **Test the Neural Language Model demo:**
   ```bash
   python3 "3 - A Neural Probabilistic Language Model/demo.py"
   ```

---

## 🤝 Contributing & Feedback

Contributions, corrections, and additional paper explanations are welcome! If you find a typo, want to suggest additional papers, or submit an implementation:
1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/new-paper`).
3. Commit your changes.
4. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
