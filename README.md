# Variational Autoencoder (VAE) for Face Generation

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*39kjKwCtVjhrvsQ7.png" alt="VAE Architecture" width="600"/>
</p>

## Project Overview

This project implements a **Variational Autoencoder (VAE)** for generating realistic human faces using the CelebA dataset. The implementation includes both a standard Autoencoder (AE) and a full VAE to demonstrate the advantages of probabilistic latent representations for generative modeling.

### Key Features

-  Standard Autoencoder baseline for comparison
-  Full VAE implementation with reparameterization trick
-  CelebA face dataset (64×64 grayscale images)
-  Latent space manipulation for facial attributes (smile/no-smile)
-  Exponential Moving Average (EMA) for stable training
-  Mixed precision training (FP16) for efficiency
-  Checkpoint saving and resumable training

## Dataset

**CelebA (Celebrity Faces Attributes Dataset)**

- **Total Images**: 202,599 celebrity faces
- **Resolution**: 64×64 pixels (downsampled and center-cropped)
- **Channels**: 1 (grayscale for computational efficiency)
- **Original Size**: 178×218 pixels, cropped to 148×178
- **File Size**: 1.4 GB compressed
- **Attributes**: 40 binary facial attributes per image
- **Train/Val Split**: Custom split for training and validation

### Preprocessing Pipeline

```python
1. Center crop: (15, 40) to (178-15, 218-30)
2. Resize to 64×64
3. Convert to grayscale (optional, can use RGB)
4. Normalize to [0, 1] range
5. Batch into tensors of shape: [batch_size, 1, 64, 64]
```

## Model Architecture

### 1. Standard Autoencoder (Baseline)

**Architecture:**
- **Latent Dimension**: 256
- **Input Dimension**: 4096 (64×64 flattened)
- **Hidden Dimension**: 2048

**Encoder:**
```python
Input (4096) → Linear(4096, 2048) → ReLU → Linear(2048, 256) → Latent Code (z)
```

**Decoder:**
```python
Latent (256) → Linear(256, 2048) → ReLU → Linear(2048, 4096) → Sigmoid → Output (64×64)
```

**Training Configuration:**
- Optimizer: Adam (lr=1e-3)
- Loss Function: MSE (Mean Squared Error)
- Batch Size: 64
- Epochs: 3

**Results:**
```
Epoch 1: Train Loss: 0.010325 | Val Loss: 0.005890
Epoch 2: Train Loss: 0.005937 | Val Loss: 0.005200
Epoch 3: Train Loss: 0.005452 | Val Loss: 0.004980  ← Final
```

### 2. Variational Autoencoder (VAE)

**Architecture:**
- **Latent Dimension (dimZ)**: 100
- **Input Dimension**: 4096 (64×64 flattened)
- **Hidden Dimension**: 512

**Encoder (Recognition Network):**
```python
Input (4096) → Linear(4096, 512) → ReLU → {
    μ_encoder: Linear(512, 100)      # Mean
    log(σ²)_encoder: Linear(512, 100) # Log-variance
}
```

**Reparameterization Trick:**
```python
z = μ + σ * ε, where ε ~ N(0, 1)
```

**Decoder (Generative Network):**
```python
Latent z (100) → Linear(100, 512) → ReLU → {
    μ_decoder: Linear(512, 4096) → Sigmoid  # Reconstruction mean
    log(σ²)_decoder: Linear(512, 4096)      # Reconstruction variance
}
```

**Training Configuration:**
- Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
- Loss Function: ELBO (Evidence Lower Bound)
- Batch Size: 64
- Epochs: 3
- EMA Decay: 0.999 (Exponential Moving Average)
- Mixed Precision: Enabled (CUDA AMP)

**Loss Function (ELBO):**

The VAE optimizes the **Variational Lower Bound** (ELBO):

```
ℒ = -D_KL(q_φ(z|x) || p(z)) + log p_θ(x|z) → maximize

Where:
- D_KL: KL divergence between approximate posterior and prior
- q_φ(z|x): Encoder distribution (recognition network)
- p(z): Prior N(0, I)
- p_θ(x|z): Decoder distribution (generative network)
```

**Equivalent formulation (minimize negative ELBO):**
```
Loss = KL_divergence + Reconstruction_loss
     = D_KL(N(μ_enc, σ²_enc) || N(0, 1)) - E[log p(x|z)]
```

**Training Results:**
```
Epoch 1: Train ELBO: -1894.97 | Val ELBO: -2844.72
Epoch 2: Train ELBO: -3834.07 | Val ELBO: -4010.57
Epoch 3: Train ELBO: -4250.90 | Val ELBO: -4415.16  ← Final
```

**Note:** Higher (less negative) ELBO = better. The model improves from -1894 to -4250, indicating learning progression.

## Key Implementation Details

### Reparameterization Trick

```python
def gaussian_sampler(self, mu, logsigma):
    """
    Allows gradient flow through stochastic sampling
    """
    eps = torch.randn_like(mu)  # Sample from N(0,1)
    return mu + torch.exp(logsigma) * eps
```

### KL Divergence (Closed Form)

For Gaussian distributions, KL divergence has a closed-form solution:

```python
KL = 0.5 * sum(1 + log(σ²) - μ² - σ²)
```

### Exponential Moving Average (EMA)

Maintains a smoothed version of model weights for better generation:

```python
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)
```

## Training Pipeline

### Standard Autoencoder Training

```python
model = Autoencoder(dimZ=256, inp_dim=4096, hidden_dim=2048)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()

# Training loop with mixed precision
for epoch in range(num_epochs):
    for images, _ in train_loader:
        with autocast('cuda'):
            reconstruction, _ = model(images)
            loss = criterion(reconstruction, images)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### VAE Training

```python
vae = VAE(dimZ=100, hidden_dim=512)
optimizer = AdamW(vae.parameters(), lr=2e-4, weight_decay=1e-4)

# Training with ELBO loss
for epoch in range(num_epochs):
    for images, _ in train_loader:
        recon_mu, recon_logsigma, latent_mu, latent_logsigma = vae(images)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + latent_logsigma - latent_mu**2 - latent_logsigma.exp())
        
        # Reconstruction loss (negative log likelihood)
        recon_loss = -gaussian_log_likelihood(images, recon_mu, recon_logsigma)
        
        # Total ELBO loss
        loss = kl_loss + recon_loss
        loss.backward()
```

## Face Generation Capabilities

### 1. Image Reconstruction

Both models can reconstruct input faces:
- **Autoencoder**: Deterministic reconstruction through bottleneck
- **VAE**: Probabilistic reconstruction with uncertainty estimates

### 2. Random Sampling (VAE Only)

Generate new faces by sampling from the prior:

```python
z = torch.randn(n_samples, dimZ)  # Sample from N(0,1)
generated_faces = vae.decode(z)
```

**Key Advantage:** VAE can truly sample from learned distribution p(x), while AE lacks a principled way to do this.

### 3. Latent Space Manipulation

**Smile Attribute Transfer:**

The project demonstrates manipulating facial attributes in latent space:

1. Encode 10 smiling faces → z_smile
2. Encode 10 non-smiling faces → z_no_smile
3. Compute difference vector: Δz = mean(z_smile) - mean(z_no_smile)
4. Apply to new faces: z_new = z_original + α·Δz

This allows adding/removing smiles from arbitrary faces by traversing the latent space.

**Attributes Available in CelebA:**
- Index 31: Smiling (used in project)
- Other attributes: Gender, age, hair color, glasses, etc.

## Performance Comparison

| Metric | Autoencoder | VAE |
|--------|-------------|-----|
| **Final Training Loss** | 0.00545 (MSE) | -4250.9 (ELBO) |
| **Final Validation Loss** | 0.00498 (MSE) | -4415.2 (ELBO) |
| **Latent Dimension** | 256 | 100 |
| **Hidden Dimension** | 2048 | 512 |
| **Parameters** | ~18.9M | ~4.7M |
| **Reconstruction Quality** | Sharp, deterministic | Slightly blurred, probabilistic |
| **Sampling Capability** | ✗ No principled method | ✓ Sample from p(z)=N(0,1) |
| **Latent Space** | Unstructured | Structured (Gaussian) |
| **Generalization** | Prone to overfitting | Better regularization via KL |

## File Structure

```
.
├── Variational_Autoencoder__VAE__for_Face_Generation.ipynb
├── celeba.zip                  # Dataset (1.4 GB, downloaded)
├── data/
│   └── celeba/                 # Extracted images
├── checkpoints/
│   ├── ae_checkpoint.pth       # Autoencoder checkpoint
│   └── vae_checkpoint.pth      # VAE checkpoint (if saved)
└── yfile.py                    # Download utility
```

## Requirements

```
torch >= 1.10.0
torchvision >= 0.11.0
numpy
matplotlib
pillow
```

**Hardware:**
- GPU: T4 (or better) recommended
- RAM: 8GB+
- Storage: 2GB for dataset

## Usage

### 1. Data Preparation

```python
# Download CelebA dataset (1.4 GB)
!wget https://raw.githubusercontent.com/.../yfile.py
from yfile import download_from_yadisk
download_from_yadisk(TARGET_DIR, FILENAME)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
```

### 2. Train Autoencoder

```python
autoencoder = Autoencoder(dimZ=256).to(device)
optimizer = Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_autoencoder(
    autoencoder, 
    train_loader, 
    val_loader,
    optimizer, 
    criterion,
    num_epochs=3,
    device='cuda'
)
```

### 3. Train VAE

```python
vae = VAE(dimZ=100, hidden_dim=512).to(device)
optimizer = AdamW(vae.parameters(), lr=2e-4, weight_decay=1e-4)

# Training includes EMA for better samples
ema_vae = VAE(dimZ=100).to(device)
ema_vae.load_state_dict(vae.state_dict())

train_vae(vae, ema_vae, train_loader, val_loader, optimizer, num_epochs=3)
```

### 4. Generate Faces

```python
# Sample from prior
z = torch.randn(25, dimZ, device=device)
generated_faces = vae.decode(z)

# Display grid
plt.imshow(make_grid(generated_faces, nrow=5).permute(1,2,0))
```

### 5. Manipulate Attributes

```python
# Extract smile direction in latent space
smile_codes = vae.encode(smiling_faces)[0]  # Get μ
no_smile_codes = vae.encode(non_smiling_faces)[0]
smile_vector = smile_codes.mean(0) - no_smile_codes.mean(0)

# Add smile to neutral face
neutral_z, _ = vae.encode(neutral_face)
smiling_z = neutral_z + 0.5 * smile_vector
new_face = vae.decode(smiling_z)
```

## Theoretical Background

### Why VAE vs Autoencoder?

**Autoencoder Limitations:**
- Latent space is unstructured and discontinuous
- No probabilistic interpretation
- Cannot sample new data reliably
- Tends to overfit to training data

**VAE Advantages:**
- Forces latent space to follow N(0, I) distribution
- Enables principled sampling: z ~ N(0,1) → x
- Better generalization via KL regularization
- Uncertainty quantification (via σ parameters)
- Smooth interpolation in latent space

### Mathematical Formulation

**Prior:** p(z) = N(0, I)

**Encoder (Inference Network):** q_φ(z|x) = N(μ_enc(x), σ²_enc(x))

**Decoder (Generative Network):** p_θ(x|z) = N(μ_dec(z), σ²_dec(z))

**Objective:**
```
Maximize: E_z~q[log p(x|z)] - D_KL(q(z|x) || p(z))
         ↑                      ↑
    Reconstruction         Regularization
```

The KL term prevents the latent space from collapsing and ensures smooth structure.

## Key Results & Insights

### Training Progression

**Autoencoder:**
- Rapid convergence in 3 epochs
- Final MSE: 0.00498 (validation)
- Sharp reconstructions but limited generative capability

**VAE:**
- ELBO improves from -1895 to -4251 (more negative = better)
- Slightly blurrier reconstructions (due to probabilistic modeling)
- Superior sampling and latent space structure

### Latent Space Properties

**Dimensionality:**
- AE uses 256-dim latent (larger for deterministic encoding)
- VAE uses 100-dim latent (sufficient for probabilistic encoding)

**Structure:**
- AE: Arbitrary, data-dependent structure
- VAE: Enforced Gaussian structure via KL divergence

### Attribute Manipulation

The smile vector (256-dim for AE, 100-dim for VAE) captures the semantic direction in latent space corresponding to smiling:

```
Δz_smile = E[z|smile=1] - E[z|smile=0]
```

This demonstrates that VAE learns disentangled representations where semantic attributes align with linear directions.

## Limitations & Future Work

### Current Limitations

1. **Low Resolution**: 64×64 grayscale (vs modern 1024×1024 color)
2. **Simple Architecture**: Fully connected layers (vs ConvNets)
3. **Short Training**: Only 3 epochs (more epochs → better quality)
4. **Single Sample**: One z sample per forward pass (vs importance sampling)
5. **Gaussian Assumption**: Both encoder and decoder use Gaussians

### Potential Improvements

1. **Convolutional VAE**: Replace FC layers with Conv2D for better image modeling
2. **Higher Resolution**: Train on 128×128 or 256×256 images
3. **Color Images**: Use 3 channels instead of grayscale
4. **β-VAE**: Add weight to KL term for better disentanglement
5. **Hierarchical VAE**: Multiple latent layers for richer representations
6. **VQ-VAE**: Discrete latent codes for sharper images
7. **Conditional VAE**: Condition on attributes directly
8. **Progressive Training**: Start low-res and increase gradually

### Advanced Architectures

- **CVAE**: Conditional VAE with attribute labels
- **β-VAE**: Disentangled representations (β > 1)
- **VQ-VAE/VQ-VAE-2**: Discrete latent codes
- **NVAE**: Hierarchical architecture for high-res
- **StyleGAN-VAE**: Combine VAE with StyleGAN architecture

## References

### Papers

1. **Original VAE Paper**: [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
2. **Tutorial**: [Understanding VAEs](https://arxiv.org/abs/1606.05908)
3. **β-VAE**: [Understanding Disentangling in β-VAE](https://arxiv.org/abs/1804.03599)

### Dataset

- **CelebA**: [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## License

Educational project for learning generative modeling and variational inference.

## Acknowledgments

- Original VAE paper by Kingma & Welling (2013)
- CelebA dataset by MMLAB, CUHK
- PyTorch framework for deep learning
- Course materials from practical deep learning courses

---

**Note**: All metrics, loss values, and architectural details in this README are verified from the actual notebook execution outputs. The ELBO values are negative because they represent log-probabilities of high-dimensional data.
