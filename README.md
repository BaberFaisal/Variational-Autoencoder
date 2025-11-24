# CelebA Variational Autoencoder (PyTorch)

Single-notebook implementation of a fully connected Autoencoder (AE) and Variational Autoencoder (VAE) on the CelebA faces dataset, including latent-space smile editing.

## Files

- `Faisal_Baber_homework_vae__.ipynb` – main notebook with all code and experiments.
- `celeba/` – dataset directory created by the notebook (not tracked).
- `real.npz` – saved real images for evaluation (created by the notebook).
- `ae_checkpoint.pth` – trained AE checkpoint (created by the notebook).

Add these to your `.gitignore`:
- `celeba/`
- `*.npz`
- `*.pth`
- `*.zip`

## Requirements

- Python 3.9+
- PyTorch (with CUDA for reasonable training time)
- torchvision
- numpy
- matplotlib
- Jupyter Notebook or JupyterLab

Install core dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

## What the notebook does

1. **Download & prepare CelebA**
   - Downloads `celeba.zip`, extracts to `celeba/celeba/img_align_celeba/`.
   - Crops and resizes faces to 64×64, converts to grayscale tensors.
   - Saves a 10k-image hold-out set to `real.npz`.

2. **Train Autoencoder (AE)**
   - Fully connected AE on 64×64 (4096-dim) inputs.
   - Loss: MSE reconstruction loss.
   - Uses mixed precision on CUDA, saves `ae_checkpoint.pth`.
   - Shows reconstructions and random samples from latent noise.

3. **Train Variational Autoencoder (VAE)**
   - Encoder outputs latent mean and log-std.
   - Decoder outputs per-pixel mean and log-std.
   - Optimizes ELBO = reconstruction log-likelihood − KL divergence.
   - Optional EMA for more stable sampling.
   - Visualizes reconstructions and random VAE samples.

4. **Latent smile editing**
   - Uses CelebA "Smiling" attribute.
   - Computes average latent code for smiling vs non-smiling faces.
   - Applies their difference to neutral faces to make them “smile”.

## How to run

1. Clone the repo and enter it:

```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```

2. (Optional) create and activate a virtual environment.

3. Install dependencies (see **Requirements**).

4. Start Jupyter and open the notebook:

```bash
jupyter notebook Faisal_Baber_homework_vae__.ipynb
```

5. Run all cells from top to bottom:
   - Data download & preparation
   - AE training + sampling
   - VAE training + sampling
   - Latent smile editing

> **Note:** The notebook assumes a CUDA GPU and uses `autocast("cuda")` and `GradScaler("cuda")`. For CPU-only runs, remove or adapt these parts before executing.
