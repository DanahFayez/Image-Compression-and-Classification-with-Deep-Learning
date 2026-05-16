# Image Compression and Deep Learning Classification

A study on how **wavelet-based lossy image compression** affects the classification accuracy of pretrained convolutional neural networks (**ResNet-18** and **MobileNetV2**) on the Mini-ImageNet dataset.

> **Course:** EE413 — Signal Processing / Deep Learning Project
> **Models:** ResNet-18, MobileNetV2 (ImageNet pretrained, fine-tuned)
> **Compression:** 2D Discrete Wavelet Transform (Haar, 2-level) with hard thresholding
> **Dataset:** Mini-ImageNet (100 classes, re-split to 500 train + 100 test per class)

---

## Table of Contents

1. [Motivation](#motivation)
2. [Background: Wavelet Compression Theory](#background-wavelet-compression-theory)
3. [Methodology](#methodology)
4. [Dataset](#dataset)
5. [Models and Hyperparameters](#models-and-hyperparameters)
6. [Results](#results)
7. [Visual Examples](#visual-examples)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [Installation and Usage](#installation-and-usage)
10. [Project Structure](#project-structure)
11. [References](#references)

---

## Motivation

Modern computer vision pipelines are often deployed in bandwidth-constrained or storage-constrained environments — mobile devices, edge sensors, satellite imagery, medical imaging archives. In all of these settings, images are **compressed** before being stored or transmitted, and a downstream classifier must then operate on the **reconstructed (lossy) version** of the image rather than the original.

This raises a natural question: **how much compression can a CNN tolerate before its classification accuracy degrades?**

This project investigates that trade-off using **wavelet-based compression** — a classical signal-processing technique that underlies the JPEG 2000 standard — and two widely used pretrained CNN architectures. By systematically varying the compression ratio and measuring accuracy at the dataset and per-class level, we aim to characterize:

1. The **robustness threshold** of CNNs to wavelet compression.
2. Which **types of image content** are most sensitive to information loss.
3. Whether the two architectures (a deep residual network and a lightweight mobile network) degrade at similar rates.

---

## Background: Wavelet Compression Theory

### Why wavelets?

Natural images contain both **smooth regions** (large uniform areas) and **localized details** (edges, textures, fine structures). Classical Fourier methods are good at representing smooth, globally periodic content but poor at localizing edges in space. **Wavelets** solve this by providing a basis that is localized in **both space and frequency**, making them well-suited to natural image signals.

A 2D Discrete Wavelet Transform (DWT) decomposes an image into:

- An **approximation sub-band** (LL): a low-resolution version capturing coarse structure.
- Three **detail sub-bands** (LH, HL, HH): horizontal, vertical, and diagonal high-frequency information.

The decomposition can be applied recursively to the LL sub-band, giving a multi-resolution pyramid. At **level** $L$, a single channel of an $N \times N$ image yields $3L + 1$ sub-bands.

Crucially, wavelets exhibit **energy compaction**: for typical natural images, most of the signal energy concentrates in a small number of large-magnitude coefficients, while most coefficients are near zero. This property is what makes wavelet compression effective — discarding the small coefficients removes very little perceptual information.

### Why the Haar wavelet?

The **Haar wavelet** is the simplest possible wavelet and was chosen for several practical reasons:

- **Orthogonality** — guarantees a well-defined inverse transform with no energy leakage between sub-bands.
- **Compact support** — the basis functions are non-zero only on a small interval, so each coefficient depends only on a local image neighborhood.
- **Computational efficiency** — Haar uses 2-tap filters, making both decomposition and reconstruction extremely fast (important for processing the full 10,000-image test set three times).
- **Interpretability** — the Haar coefficients correspond to simple averages and differences of neighboring pixels, making the effect of thresholding easy to reason about.

Smoother wavelets (e.g. `db4`, `sym8`, `bior4.4`) generally give better perceptual quality at the same compression ratio, but Haar provides a clean and aggressive baseline that exposes the failure modes of compression most clearly — which is exactly what we want when studying classifier robustness.

### Why hard thresholding?

Given the wavelet coefficient array $W$ of a single channel, the compression operator applied in this project is **top-$K$ hard thresholding**:

```math
W'_i = \begin{cases} W_i & \text{if } |W_i| \text{ is among the } K \text{ largest values in } |W| \\ 0 & \text{otherwise} \end{cases}
```

where the number of retained coefficients is

```math
K = \left\lfloor \frac{N}{r} \right\rfloor
```

with $N$ the total number of coefficients and $r \in \{2, 5, 10\}$ the compression ratio. The compressed image is then reconstructed by the inverse transform:

```math
\hat{I} = \mathcal{W}^{-1}(W')
```

Hard thresholding was chosen over soft thresholding because:

- It **preserves the magnitude** of the retained coefficients exactly, so the kept information is undistorted.
- It gives **direct control** over the compression ratio — keeping exactly $N/r$ coefficients yields an effective $r{:}1$ ratio in the coefficient domain.
- It produces a **clean isolation** of the variable under study (number of coefficients kept), without introducing a separate shrinkage parameter.

The full per-channel pipeline can be written compactly as:

```math
\hat{I}_c = \mathcal{W}^{-1}\left( T_K\left( \mathcal{W}(I_c) \right) \right), \quad c \in \{R, G, B\}
```

where $\mathcal{W}$ is a 2-level Haar 2D DWT and $T_K$ is the top-$K$ hard-thresholding operator.

---

## Methodology

The full experimental pipeline consists of nine stages, each implemented as a self-contained section of `imagecompression.py`.

### Stage 1 — Data acquisition

The Mini-ImageNet archive files (`train.tar`, `val.tar`, `test.tar`) are copied from a Google Drive folder into the Colab runtime. Local copying is performed once at the start because reading directly from mounted Drive during training is significantly slower than reading from the local filesystem.

### Stage 2 — Extraction and verification

Each `.tar` is extracted into `/content/miniimagenet_extracted/{train,val,test}/`. A verification step counts the number of class folders and images per split to confirm that extraction succeeded and that no class is empty.

### Stage 3 — Re-splitting into a classification dataset

Mini-ImageNet was originally designed for **few-shot learning**, with 64 training classes, 16 validation classes, and 20 test classes — i.e. **disjoint class sets**. This is unsuitable for standard supervised classification, where every class must appear in both training and test sets.

The dataset is therefore re-organized:

1. Images from all three original splits are pooled per class.
2. Each class is **shuffled with a fixed seed** (`seed = 42`) for reproducibility.
3. The first **500 images** are assigned to the new training set and the next **100** to the new test set.

This yields a balanced dataset of **100 classes × 600 images = 60,000 images** (50,000 train / 10,000 test).

### Stage 4 — Model setup (transfer learning)

Both ResNet-18 and MobileNetV2 are loaded with their **ImageNet-pretrained weights**, and the final fully-connected classifier is replaced with a new linear layer of output dimension 100. This is standard **transfer learning**: the convolutional feature extractor is initialized from a model that already understands generic visual primitives (edges, textures, shapes), and fine-tuning adapts these features to the Mini-ImageNet domain.

All layers are left **trainable** (no freezing). This was chosen because the target domain (Mini-ImageNet) is a strict subset of the source domain (ImageNet), so the pretrained features are already well-aligned and unlikely to be destroyed by fine-tuning.

### Stage 5 — Training and validation

A **10 %** (ResNet-18) or **20 %** (MobileNetV2) hold-out is split from the training set as a validation set, using a fixed seed for reproducibility. Training proceeds with:

- **Loss:** cross-entropy
- **Optimizer:** Adam with weight decay
- **Augmentation:** random horizontal flip on the training subset only
- **Normalization:** ImageNet mean and standard deviation
- **Resolution:** 96 × 96 (chosen to balance speed and accuracy on a Colab GPU)

After each epoch, the model is evaluated on the validation set. The **best checkpoint** (highest validation accuracy) is saved to disk — rather than the final epoch — because the training accuracy continued to rise after validation accuracy had plateaued, indicating mild overfitting.

### Stage 6 — Baseline evaluation

Each best-checkpoint model is evaluated on the **uncompressed** test set, producing the **baseline accuracy** used as the reference point for all subsequent compression experiments.

### Stage 7 — Generating compressed test sets

For each compression ratio $r \in \{2, 5, 10\}$, the wavelet compression operator is applied **image by image** to the original test set, and the reconstructed images are saved to `compressed_test/ratio_{r}_1/` with the original class folder structure preserved. This produces three parallel test sets that are class-aligned with the original — only the pixel content is altered.

Compression is applied **once** and saved to disk so that evaluation is deterministic and reproducible.

### Stage 8 — Compressed-set evaluation

Each trained model is evaluated on each of the three compressed test sets using the **same preprocessing transforms** as the original test set (resize, tensor conversion, ImageNet normalization — no augmentation). The classification accuracy is recorded and the **accuracy drop** relative to the baseline is computed.

### Stage 9 — Per-class sensitivity analysis

For the most aggressive ratio (10:1), per-class accuracy is computed on both the original and the compressed test set. The classes are then sorted by **accuracy drop** to identify which categories are most vulnerable to information loss. The top 10 most affected classes are reported and visualized.

---

## Dataset

**Mini-ImageNet** — a 100-class subset of ImageNet.

| Split            | Classes | Images / class | Total  |
|------------------|---------|----------------|--------|
| Train (new)      | 100     | 500            | 50,000 |
| Test (new)       | 100     | 100            | 10,000 |
| Validation       | (held out from train) | varies | 5,000–10,000 |

All splits use a fixed random seed (`42`).

---

## Models and Hyperparameters

| Hyperparameter      | ResNet-18           | MobileNetV2         |
|---------------------|---------------------|---------------------|
| Pretrained weights  | ImageNet            | ImageNet            |
| Output classes      | 100                 | 100                 |
| Input resolution    | 96 × 96             | 96 × 96             |
| Batch size          | 64                  | 64                  |
| Optimizer           | Adam                | Adam                |
| Learning rate       | 1 × 10⁻⁴            | 1 × 10⁻⁴            |
| Weight decay        | 1 × 10⁻⁴            | —                   |
| Loss                | Cross-entropy       | Cross-entropy       |
| Epochs              | 5                   | 3                   |
| Train/val split     | 90 / 10             | 80 / 20             |
| Augmentation        | Random horizontal flip | Random horizontal flip |
| Device              | CUDA GPU (Colab)    | CUDA GPU (Colab)    |

Checkpoint selection: **best validation accuracy** across all epochs.

---

## Results

### Accuracy on original and wavelet-compressed test sets

| Test Set        | ResNet-18              | MobileNetV2            |
|-----------------|------------------------|------------------------|
| Original        | **71.50 %** (baseline) | **75.15 %** (baseline) |
| 2:1 compressed  | **71.25 %**            | **74.85 %**            |
| 5:1 compressed  | **68.73 %**            | **68.84 %**            |
| 10:1 compressed | **50.70 %**            | **48.23 %**            |

### Accuracy drop relative to baseline

| Compression ratio | ResNet-18 drop | MobileNetV2 drop |
|-------------------|----------------|------------------|
| 2:1               | − 0.25 pp       | −0.21 pp         |
| 5:1               | −2.77 pp       | −7.08 pp         |
| 10:1              | −21.00 pp      | −27.82 pp        |

*(pp = percentage points)*

### Discussion

Several observations stand out:

1. **Mild compression is essentially free.** At 2:1, both models lose less than one percentage point of accuracy (ResNet-18: −0.25 pp, MobileNetV2: −0.21 pp). Keeping the top 50 % of wavelet coefficients preserves enough information that the classifiers are effectively unaffected. This is consistent with the energy-compaction property of the wavelet transform — half of the coefficients carry the vast majority of the signal energy in natural images.

2. **Moderate compression begins to bite.** At 5:1, accuracy drops by 4–6 pp. Interestingly, **MobileNetV2 degrades faster than ResNet-18 here** (−6.31 pp vs −2.77 pp), despite having a slightly higher baseline. This suggests that the lightweight depth-wise-separable convolutions in MobileNetV2 may rely more heavily on mid-frequency texture cues that are among the first to be zeroed out by hard thresholding.

3. **Aggressive compression is catastrophic.** At 10:1, both models lose roughly **a quarter of their accuracy in absolute terms** (ResNet-18: −21.00 pp, MobileNetV2: −26.92 pp). Keeping only 10 % of wavelet coefficients removes enough high- and mid-frequency content that many class-discriminative features disappear, and accuracy approaches the level where the classifier is no longer competitive.

4. **The degradation curve is highly non-linear.** Moving from 2:1 to 5:1 costs a few percentage points; moving from 5:1 to 10:1 costs roughly five times as much. The dominant failure regime sits between these two operating points, which is exactly the range where practical wavelet codecs typically operate — suggesting that classifier-aware rate selection is important.

5. **MobileNetV2 is less robust than ResNet-18.** Although MobileNetV2 wins on the original test set, it loses that lead at 5:1 and falls **below** ResNet-18 at 10:1. For deployment scenarios where input images are likely to be heavily compressed, the deeper but still-modest ResNet-18 appears to be the safer choice on this dataset.

### Per-class sensitivity (10:1 compression)

The per-class analysis identifies the **top 10 classes with the largest accuracy drop** between the original and 10:1 compressed test sets. Visual inspection of sample images from these classes suggests that **fine textures, repetitive patterns, and subtle inter-class features** are the most fragile — exactly the content that lives in the high-frequency wavelet sub-bands that hard thresholding discards first.

---

## Visual Examples

><img width="1331" height="389" alt="image" src="https://github.com/user-attachments/assets/cd113762-073b-4103-9818-29b7ec392291" />
:_

**Figure 1 — Compression at different ratios on a sample image**


A 1 × 4 panel showing the original image alongside its 2:1, 5:1, and 10:1 reconstructions. As $r$ increases, fine textures fade first, then mid-frequency content (edges of small objects), with the coarse structure preserved even at 10:1.

**Figure 2 — Accuracy vs compression ratio**

><img width="858" height="553" alt="image" src="https://github.com/user-attachments/assets/62766526-1c15-4a67-acb8-3b47e9e9ed37" />
>


A bar chart of classification accuracy at {Original, 2:1, 5:1, 10:1}. Used to compare degradation curves between ResNet-18 and MobileNetV2.

**Figure 3 — Top 10 classes most affected by 10:1 compression**

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/8b9671ec-c760-4dc9-8b77-85feba81ce6e" />

A bar chart of accuracy drop per class, sorted descending. Used together with the sample-image montage below to interpret *what kind of visual content* is most vulnerable.


---

## Limitations and Future Work

### Limitations

1. **Single wavelet family.** Only the Haar wavelet was evaluated. Smoother wavelets (Daubechies, Symlets, biorthogonal) are known to yield better reconstruction quality at the same coefficient-retention rate and would likely shift the accuracy-vs-compression curve.
2. **Fixed decomposition level.** All experiments use a 2-level DWT. Deeper decompositions concentrate more energy in the approximation sub-band and may behave differently under thresholding.
3. **Hard thresholding only.** Soft thresholding and uniform scalar quantization (as used in real codecs like JPEG 2000) were not compared. The "compression ratio" reported here is therefore a **coefficient-domain** ratio, not a true bitstream compression ratio.
4. **Channel-independent compression.** R, G, B are compressed separately, which is suboptimal — real codecs operate in a decorrelated colour space such as YCbCr and allocate more coefficients to luminance.
5. **Low input resolution.** Images are resized to 96 × 96 before classification, which itself acts as a low-pass filter and may mask some of the effects of compression at higher resolutions.
6. **Limited training budget.** ResNet-18 was trained for 5 epochs and MobileNetV2 for 3. Longer training with a learning-rate schedule would likely raise the baseline accuracy and may affect robustness to compression.
7. **No comparison to a standard codec.** A JPEG-quality-factor sweep was not included as a control, so the results cannot be directly compared to industry-standard compression.

### Future Work

- Repeat the experiment with **`db4`, `sym4`, and `bior4.4`** wavelets and report the Pareto front of (compression ratio, accuracy).
- Compare against **JPEG** and **JPEG 2000** at matched file sizes to ground the wavelet results in a real-world baseline.
- Apply compression in a **decorrelated colour space** (YCbCr with chroma sub-sampling) to better model practical codecs.
- Run **compression-aware training** — train the model on a mixture of original and compressed images and measure whether robustness improves.
- Extend the per-class analysis with **Grad-CAM** or similar attribution methods to localize *which image regions* drive the accuracy drop on the most affected classes.
- Evaluate at the **full ImageNet resolution (224 × 224)** to see whether the conclusions hold at higher input fidelity.

---

## Installation and Usage

### Dependencies

```bash
pip install torch torchvision PyWavelets numpy matplotlib pillow tqdm gdown
```

### Option 1 — Google Colab (recommended)

The script was developed in Colab with GPU runtime. Open it as a notebook, enable GPU, ensure the Mini-ImageNet `.tar` files are in your Google Drive at the path defined in the script, and run the cells in order.

### Option 2 — Local execution

```bash
python imagecompression.py
```

The script contains Colab-specific magic commands (`!pip install`, `!gdown`, `drive.mount(...)`). These must be removed or adapted, and the dataset paths updated to point to your local Mini-ImageNet copy.

---

## Project Structure

```
.
├── imagecompression.py              # Main script (exported from Colab notebook)
├── README.md                        # This file
├── best_resnet18_miniimagenet.pth   # Best ResNet-18 checkpoint (generated)
└── figures/                         # Generated plots (optional)
    ├── compression_comparison.png
    ├── accuracy_vs_compression.png
    ├── top10_affected_classes.png
    └── most_affected_samples.png
```

The pipeline inside `imagecompression.py` is organized as:

1. Installation & imports
2. Dataset download & extraction
3. Re-splitting into 100-class classification format
4. ResNet-18 fine-tuning and baseline evaluation
5. Wavelet compression of the test set at 2:1, 5:1, 10:1
6. ResNet-18 evaluation on compressed sets
7. MobileNetV2 fine-tuning and baseline
8. MobileNetV2 evaluation on compressed sets
9. Per-class sensitivity analysis

---

## References

- Mallat, S. *A Wavelet Tour of Signal Processing*, 3rd ed., Academic Press, 2008.
- Haar, A. "Zur Theorie der orthogonalen Funktionensysteme," *Mathematische Annalen*, 1910.
- He, K. et al. "Deep Residual Learning for Image Recognition," *CVPR*, 2016. (ResNet)
- Sandler, M. et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.
- Vinyals, O. et al. "Matching Networks for One Shot Learning," *NeurIPS*, 2016. (Mini-ImageNet)
- Lee, G. et al. "PyWavelets: A Python package for wavelet analysis," *Journal of Open Source Software*, 2019.

---

## Acknowledgments

- Mini-ImageNet dataset (subset of ImageNet, ILSVRC).
- PyTorch model zoo for pretrained ResNet-18 and MobileNetV2 weights.
- The PyWavelets library for the wavelet transform implementation.
