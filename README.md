# Variational Autoencoder (VAE) for Face Generation

This repository contains a PyTorch-based implementation of a Variational Autoencoder (VAE) trained on the CelebA dataset to generate and reconstruct human faces.

##  Overview

The project follows the seminal work of Kingma & Welling (2013) to build a generative model capable of learning a structured latent representation of human facial features. 

**Key Features:**
- **Architecture**: Deep Convolutional VAE.
- **Latent Space**: Structured Gaussian latent space using the reparameterization trick.
- **Dataset**: CelebA (Celebrity Faces Attributes Dataset).
- **Optimization**: Evidence Lower Bound (ELBO) maximization.

## Technical Implementation

### 1. Data Pipeline
- Automated downloading and extraction of the CelebA dataset.
- Image preprocessing and normalization for neural network compatibility.

### 2. Model Architecture
- **Encoder**: Compresses input images into latent mean and variance vectors.
- **Decoder**: Resamples from the latent space to reconstruct the input or generate novel faces.
- **Reparameterization**: Enables gradient descent on stochastic layers.

### 3. Training Logic
- Loss function: `Reconstruction Loss + KL Divergence`.
- Hyperparameter tuning for latent dimension size and learning rates.

##  Requirements

To run this notebook, ensure you have the following installed:
- Python 3.x
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- TQDM

```bash
pip install torch torchvision numpy matplotlib tqdm
```


## 📜 References
- Kingma, D. P., & Welling, M. (2013). [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114).
