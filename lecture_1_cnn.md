# Convolutional Neural Networks (CNNs)

## Introduction

Feedforward networks treat each input feature independently, with no built-in notion of spatial structure. For image data, this is a severe mismatch: a pixel's meaning depends critically on its neighbors, and the same pattern — an edge, a corner, a texture — carries information regardless of where in the image it appears.

**Convolutional Neural Networks** exploit this structure through two core ideas: **local connectivity**, so each neuron responds only to a small spatial neighborhood, and **weight sharing**, so the same filter is applied across the entire input. The result is an architecture that is dramatically more parameter-efficient than a fully connected network on images, and that builds representations naturally aligned with how visual information is organized.

This lecture develops the mathematical foundations of CNNs, their key components, training considerations, and practical limitations.

---

## 1. The Problem with Fully Connected Networks on Images

### Dimensionality

Consider a modest 224×224 RGB image — a standard input size. Flattened, this is a vector of $224 \times 224 \times 3 = 150{,}528$ values. A single fully connected hidden layer with 1,000 neurons would require:

$$150{,}528 \times 1{,}000 = 150{,}528{,}000 \text{ parameters}$$

For the first layer alone. This is computationally intractable for deeper networks and leads to severe overfitting on typical dataset sizes.

### Ignored Structure

Beyond the parameter count, a fully connected layer treats pixel $(0, 0)$ and pixel $(100, 100)$ as entirely unrelated features. It has no mechanism to recognize that adjacent pixels tend to be correlated, or that a diagonal edge in the top-left corner and the same edge in the bottom-right corner are the same pattern.

CNNs address both problems simultaneously.

---

## 2. The Convolution Operation

### Definition

A **convolutional layer** applies a learned filter $W \in \mathbb{R}^{k \times k \times C_{in}}$ to an input feature map $X \in \mathbb{R}^{H \times W \times C_{in}}$. The output at spatial position $(i, j)$ for a single filter is:

$$Z_{i,j} = \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W_{u,v,c} \cdot X_{i+u,\, j+v,\, c} + b$$

where $k$ is the **kernel size**, $C_{in}$ is the number of input channels, and $b$ is a scalar bias.

With $C_{out}$ filters, the output is a feature map of shape $\mathbb{R}^{H' \times W' \times C_{out}}$, where each of the $C_{out}$ channels corresponds to a different learned filter.

### Key Properties

**Local connectivity**: Each output unit $Z_{i,j}$ depends only on a $k \times k$ spatial region of the input — its **receptive field**. For a 3×3 kernel, each output unit sees 9 input positions per channel.

**Weight sharing**: The same filter $W$ is applied at every spatial position $(i, j)$. This means the network learns a pattern detector that fires wherever the pattern occurs, regardless of location — a property called **translation equivariance**.

**Parameter count**: A convolutional layer with $C_{in}$ input channels, $C_{out}$ filters, and kernel size $k$ has:

$$|\theta_{\text{conv}}| = C_{out} \cdot (k^2 \cdot C_{in} + 1)$$

For a 3×3 conv with 32 input channels and 64 output filters: $64 \times (9 \times 32 + 1) = 18{,}496$ parameters — orders of magnitude fewer than a fully connected equivalent.

---

## 3. Padding and Stride

### Padding

Without padding, each convolution reduces the spatial dimensions. For a $k \times k$ kernel applied to an $H \times W$ input:

$$H_{\text{out}} = H - k + 1, \quad W_{\text{out}} = W - k + 1$$

**Zero padding** adds $p$ rows/columns of zeros around the input boundary. With $p = \lfloor k/2 \rfloor$ (same padding), the output has the same spatial dimensions as the input:

$$H_{\text{out}} = H - k + 2p + 1 = H \quad \text{when } p = \lfloor k/2 \rfloor$$

This is the standard choice in modern architectures.

### Stride

The **stride** $s$ controls how far the filter moves between applications:

$$H_{\text{out}} = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1$$

Stride $s = 1$ is standard. Stride $s = 2$ halves the spatial dimensions, serving a similar purpose to pooling while keeping the operation learnable.

---

## 4. Pooling

Pooling layers reduce spatial dimensions, providing:
- **Computational efficiency**: smaller feature maps downstream
- **Translation invariance**: small shifts in the input produce the same pooled output
- **Increased receptive field**: later layers see a larger portion of the original input

### Max Pooling

$$\text{MaxPool}(X)_{i,j} = \max_{u,v \in \mathcal{R}_{i,j}} X_{u,v}$$

Takes the maximum value in each pooling window $\mathcal{R}_{i,j}$. A 2×2 max pool with stride 2 halves each spatial dimension:

$$H_{\text{out}} = \frac{H}{2}, \quad W_{\text{out}} = \frac{W}{2}$$

Max pooling is the standard choice — it retains the strongest activation in each region, preserving the presence of detected features.

### Average Pooling

Takes the mean instead of the maximum. Used in some architectures (e.g., global average pooling before the classifier head) but less common than max pooling in intermediate layers.

---

## 5. CNN Architecture

### The Standard Block

The fundamental building block of a CNN is:

$$\text{Conv2d} \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}$$

Each block increases the number of channels while reducing spatial dimensions. A typical progression for a 28×28 grayscale input:

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (1, 28, 28) | — |
| Conv(1→32, 3×3, pad=1) + BN + ReLU | (32, 28, 28) | 320 |
| MaxPool(2×2) | (32, 14, 14) | — |
| Conv(32→64, 3×3, pad=1) + BN + ReLU | (64, 14, 14) | 18,496 |
| MaxPool(2×2) | (64, 7, 7) | — |
| Flatten | 3,136 | — |
| Linear(3136→256) + BN + ReLU + Dropout | 256 | 802,048 |
| Linear(256→10) | 10 | 2,570 |

### Classifier Head

After the convolutional blocks, a `Flatten` layer converts the 3D feature map to a 1D vector, which feeds into one or more fully connected layers:

$$\text{Flatten} \to \text{Linear} \to \text{BN} \to \text{ReLU} \to \text{Dropout} \to \text{Linear} \to \text{Logits}$$

The final linear layer outputs raw **logits** — one per class. No softmax is applied here; PyTorch's `CrossEntropyLoss` combines log-softmax and NLL loss in a single numerically stable operation.

### Receptive Field Growth

Each convolutional layer increases the **effective receptive field** — the region of the original input that influences a single output unit. For $L$ convolutional layers with kernel size $k$:

$$\text{RF}_L = 1 + L \cdot (k - 1)$$

With pooling layers, the receptive field grows faster. Deep networks can have receptive fields that span the entire input, allowing late layers to integrate global context.

---

## 6. Batch Normalization in CNNs

`BatchNorm2d` normalizes each channel across the batch and spatial dimensions. For a feature map $Z \in \mathbb{R}^{N \times C \times H \times W}$, for each channel $c$:

$$\mu_c = \frac{1}{NHW} \sum_{n,i,j} Z_{n,c,i,j}, \qquad \sigma_c^2 = \frac{1}{NHW} \sum_{n,i,j} (Z_{n,c,i,j} - \mu_c)^2$$

$$\hat{Z}_{n,c,i,j} = \frac{Z_{n,c,i,j} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad \tilde{Z}_{n,c,i,j} = \gamma_c \hat{Z}_{n,c,i,j} + \beta_c$$

The learnable parameters $\gamma_c$ and $\beta_c$ are per-channel — one pair per output channel of the preceding conv layer, which is why `BatchNorm2d(C)` must always receive the same `C` as `out_channels` of the preceding `Conv2d`.

Benefits in convolutional architectures: allows higher learning rates, reduces sensitivity to initialization, and acts as a mild regularizer through the stochasticity of batch statistics.

---

## 7. Implementation in PyTorch

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # must match out_channels above
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    # 28x28 -> 14x14
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),   # must match out_channels above
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)    # raw logits
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


# Training setup
model     = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()          # expects raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
```

### Key Design Rules

- `BatchNorm2d(C)` always follows `Conv2d(..., out_channels=C, ...)` — C must match
- `BatchNorm1d(F)` always follows `Linear(..., out_features=F, ...)` — F must match
- No activation on the final linear layer — `CrossEntropyLoss` handles it
- Dropout goes between the penultimate and final linear layers only

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `out_channels` | 32–512 | Number of filters per layer, increases with depth |
| `kernel_size` | 3 or 5 | Larger kernels see more context, cost more compute |
| `padding` | `kernel_size // 2` | Preserves spatial dimensions |
| `dropout` | 0.3–0.5 | Regularization in the classifier head |
| `weight_decay` | 1e-4 to 1e-3 | L2 regularization on all parameters |

---

## 8. What CNNs Learn

The representations learned by a CNN follow a consistent hierarchical structure, observable by inspecting filter weights and feature maps:

**Conv Block 1 — primitive detectors**: The first layer learns edge detectors (sharp dark-to-light transitions), gradient detectors (smooth intensity ramps), and texture detectors (high-contrast local patterns). This is a universal result — virtually every CNN trained on images rediscovers these primitives, regardless of the dataset.

**Conv Block 2 — composite detectors**: Filters in deeper layers combine the primitives from the first layer into more complex patterns — curves, junctions, simple shapes. These filters operate on 32-channel feature maps rather than raw pixels, so direct visual interpretation of their weights becomes difficult.

**Classifier head — global discrimination**: The fully connected layers combine spatial evidence from across the feature maps to make a final class decision. By the time activations reach this stage, spatial information has been largely discarded — what remains is a distributed representation of which features were present and how strongly.

---

## 9. Data Augmentation

CNNs have many parameters and can overfit, especially on smaller datasets. **Data augmentation** applies random transformations to training images, effectively multiplying the dataset size and improving generalization:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # 50% chance of flipping
    transforms.RandomCrop(28, padding=2),    # random crop with 2px padding
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.286], std=[0.353])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.286], std=[0.353])
])
```

**Critical rule**: augmentation is applied only to the training set. Validation and test sets use only normalization — augmenting them would give an inconsistent measure of true performance.

Common augmentations for image classification:

| Augmentation | Use case |
|---|---|
| `RandomHorizontalFlip` | Most natural images, clothing |
| `RandomCrop` | Teaches position invariance |
| `ColorJitter` | RGB datasets where lighting varies |
| `RandomRotation` | When orientation is not class-defining |
| `RandomErasing` | Forces reliance on multiple regions |

---

## 10. Strengths and Weaknesses

### Strengths

**Parameter efficiency**: Weight sharing across spatial positions means a 3×3 conv layer with 64 filters has ~18K parameters regardless of the input image size. A fully connected equivalent on a 224×224 image would require hundreds of millions.

**Translation equivariance**: The same filter applied everywhere means the network detects a pattern in the top-left corner just as well as the bottom-right. This inductive bias is well-matched to natural images where object position should not affect recognition.

**Hierarchical feature learning**: The sequential composition of conv blocks naturally builds representations from low-level edges to high-level object parts, mirroring the structure of biological visual systems.

**Strong empirical performance**: CNNs dominated image classification, object detection, and segmentation benchmarks for over a decade and remain extremely competitive, especially on moderate-sized datasets where Vision Transformers overfit.

**Interpretability**: The first layer's filters and intermediate feature maps are directly visualizable, giving more insight into learned representations than most other deep architectures.

### Weaknesses

**Limited global context**: Standard convolutions are local operations. A 3×3 kernel at layer 1 only sees a 3×3 patch. Reaching global context requires many layers or large kernels, making it hard to capture long-range spatial dependencies without very deep architectures.

**Translation equivariance ≠ invariance**: CNNs are equivariant to translation (the response moves with the pattern) but not invariant — pooling introduces approximate invariance but a shifted image will still produce a different feature map. This is why data augmentation is important.

**Not rotation or scale invariant**: A CNN that recognizes a horizontal edge will not automatically recognize the same edge rotated 45° unless it was trained on rotated examples. Invariance to these transformations must be learned from data or explicitly engineered.

**Quadratic cost with image size**: Convolutional operations scale as $O(H \cdot W \cdot k^2 \cdot C_{in} \cdot C_{out})$. For high-resolution inputs this becomes expensive, which is why downsampling early (aggressive pooling or strided convolutions) is standard.

**Superseded for very large datasets**: Vision Transformers (ViT) have matched or exceeded CNNs on large-scale benchmarks (ImageNet-21k, JFT). However, CNNs remain the default on smaller datasets where the convolutional inductive bias provides a data-efficiency advantage.

---

## 11. Summary

1. **CNNs replace global connectivity with local filters applied via convolution**, drastically reducing parameters while exploiting the spatial structure of image data through weight sharing and local receptive fields.

2. **The convolution operation** slides a learned filter across the input, computing dot products at each position. With $C_{out}$ filters, the output has $C_{out}$ channels — each a spatial map of where that filter's pattern was detected.

3. **Padding preserves spatial dimensions**; **pooling reduces them** while increasing translation invariance and expanding the effective receptive field of deeper layers.

4. **BatchNorm2d** normalizes each channel independently across the batch and spatial dimensions. Its argument must always match the `out_channels` of the preceding `Conv2d`.

5. **CNNs learn hierarchically**: primitive edge and texture detectors in early layers, composite shape detectors in later layers, global class discriminators in the fully connected head.

6. **Data augmentation** is essential for generalization and must be applied only to the training set.

7. **CNNs are the default architecture for image classification on moderate-sized datasets**. For very large datasets, Vision Transformers are competitive, but the CNN's inductive bias remains valuable when data is limited.

---

## References

- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*.

---

**Previous Lecture**: Weighted Loss for Imbalanced Classification  
**Next Lecture**: Recurrent Neural Networks (RNN)
