"""
Problem Set 2 - Question 2: Model Comparison & Quantitative Analysis
Compares Autoencoder vs VAE with SSIM and volume fraction metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("Loading analysis tools...")

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# =============================================================================
# MODEL DEFINITIONS (same as training scripts)
# =============================================================================

class ImprovedAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ImprovedAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_encode = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.decoder(x)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =============================================================================
# METRICS
# =============================================================================

def calculate_ssim(original, reconstructed):
    """Calculate SSIM between original and reconstructed images."""
    ssim_scores = []
    for orig, recon in zip(original, reconstructed):
        # Ensure binary
        orig_bin = (orig > 0.5).astype(np.float32)
        recon_bin = (recon > 0.5).astype(np.float32)
        score = ssim(orig_bin, recon_bin, data_range=1.0)
        ssim_scores.append(score)
    return np.array(ssim_scores)


def calculate_volume_fraction(images):
    """Calculate volume fraction (proportion of white pixels)."""
    fractions = []
    for img in images:
        img_bin = (img > 0.5).astype(np.float32)
        fraction = img_bin.mean()
        fractions.append(fraction)
    return np.array(fractions)


def calculate_connectivity(image):
    """Simple connectivity measure: average cluster size."""
    from scipy import ndimage
    img_bin = (image > 0.5).astype(np.int32)
    labeled, num_features = ndimage.label(img_bin)
    if num_features == 0:
        return 0
    sizes = ndimage.sum(img_bin, labeled, range(1, num_features + 1))
    return np.mean(sizes) if len(sizes) > 0 else 0


# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

def load_data(npz_path, target_size=64):
    """Load and prepare data."""
    data = np.load(npz_path, allow_pickle=True)
    images = data['X']
    
    resized = []
    for img in images:
        if img.max() <= 1.0:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)
        
        pil_img = Image.fromarray(img_uint8)
        pil_resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
        resized.append(np.array(pil_resized))
    
    images_resized = np.array(resized)
    images_norm = images_resized.astype(np.float32) / 255.0
    images_norm = (images_norm > 0.5).astype(np.float32)
    
    return images_norm


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    NPZ_PATH = "grf_data/MICRO2D_GRF_1k.npz"
    
    print("="*60)
    print("QUANTITATIVE ANALYSIS: Autoencoder vs VAE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    images = load_data(NPZ_PATH)
    print(f"Loaded {len(images)} images")
    
    # Load models
    print("\nLoading trained models...")
    
    ae_model = ImprovedAutoencoder(latent_dim=64)
    vae_model = VAE(latent_dim=128)
    
    try:
        ae_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
        print("✓ Loaded autoencoder_model.pth")
        ae_loaded = True
    except:
        print("✗ Could not load autoencoder_model.pth - will retrain")
        ae_loaded = False
    
    try:
        vae_model.load_state_dict(torch.load("vae_model_v2.pth", map_location=device))
        print("✓ Loaded vae_model_v2.pth")
        vae_loaded = True
    except:
        print("✗ Could not load vae_model_v2.pth - will retrain")
        vae_loaded = False
    
    ae_model = ae_model.to(device)
    vae_model = vae_model.to(device)
    ae_model.eval()
    vae_model.eval()
    
    # Generate reconstructions and samples
    print("\nGenerating reconstructions...")
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(images).unsqueeze(1).to(device)
        
        # Autoencoder reconstructions
        ae_recon, ae_latent = ae_model(test_tensor)
        ae_recon = ae_recon.cpu().numpy()[:, 0]
        ae_latent = ae_latent.cpu().numpy()
        
        # VAE reconstructions
        vae_recon, vae_mu, vae_logvar = vae_model(test_tensor)
        vae_recon = vae_recon.cpu().numpy()[:, 0]
        vae_latent = vae_mu.cpu().numpy()
        
        # Generate new samples
        # Autoencoder: sample from learned distribution
        ae_mean = ae_latent.mean(axis=0)
        ae_std = ae_latent.std(axis=0)
        ae_random = np.random.randn(100, 64) * ae_std + ae_mean
        ae_generated = ae_model.decode(torch.FloatTensor(ae_random).to(device)).cpu().numpy()[:, 0]
        
        # VAE: sample from learned distribution
        vae_mean = vae_latent.mean(axis=0)
        vae_std = vae_latent.std(axis=0)
        vae_random = np.random.randn(100, 128) * vae_std + vae_mean
        vae_generated = vae_model.decode(torch.FloatTensor(vae_random).to(device)).cpu().numpy()[:, 0]
    
    # =============================================================================
    # CALCULATE METRICS
    # =============================================================================
    
    print("\n" + "="*60)
    print("QUANTITATIVE METRICS")
    print("="*60)
    
    # SSIM scores
    ae_ssim = calculate_ssim(images, ae_recon)
    vae_ssim = calculate_ssim(images, vae_recon)
    
    print(f"\n📊 SSIM (Structural Similarity) - Higher is better:")
    print(f"   Autoencoder: {ae_ssim.mean():.4f} ± {ae_ssim.std():.4f}")
    print(f"   VAE:         {vae_ssim.mean():.4f} ± {vae_ssim.std():.4f}")
    
    # Volume fractions
    real_vf = calculate_volume_fraction(images)
    ae_recon_vf = calculate_volume_fraction(ae_recon)
    vae_recon_vf = calculate_volume_fraction(vae_recon)
    ae_gen_vf = calculate_volume_fraction(ae_generated)
    vae_gen_vf = calculate_volume_fraction(vae_generated)
    
    print(f"\n📊 Volume Fraction (proportion of white phase):")
    print(f"   Real images:        {real_vf.mean():.4f} ± {real_vf.std():.4f}")
    print(f"   AE reconstructed:   {ae_recon_vf.mean():.4f} ± {ae_recon_vf.std():.4f}")
    print(f"   VAE reconstructed:  {vae_recon_vf.mean():.4f} ± {vae_recon_vf.std():.4f}")
    print(f"   AE generated:       {ae_gen_vf.mean():.4f} ± {ae_gen_vf.std():.4f}")
    print(f"   VAE generated:      {vae_gen_vf.mean():.4f} ± {vae_gen_vf.std():.4f}")
    
    # MSE
    ae_mse = np.mean((images - (ae_recon > 0.5).astype(np.float32))**2)
    vae_mse = np.mean((images - (vae_recon > 0.5).astype(np.float32))**2)
    
    print(f"\n📊 MSE (Mean Squared Error) - Lower is better:")
    print(f"   Autoencoder: {ae_mse:.6f}")
    print(f"   VAE:         {vae_mse:.6f}")
    
    # =============================================================================
    # CREATE COMPARISON FIGURES
    # =============================================================================
    
    print("\nCreating comparison figures...")
    
    # Figure 1: Side-by-side reconstruction comparison
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle("Model Comparison: Original vs Autoencoder vs VAE Reconstructions", fontsize=14)
    
    for i in range(6):
        # Original
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Autoencoder
        axes[1, i].imshow((ae_recon[i] > 0.5).astype(np.float32), cmap='gray')
        axes[1, i].set_title(f"SSIM={ae_ssim[i]:.3f}")
        axes[1, i].axis('off')
        
        # VAE
        axes[2, i].imshow((vae_recon[i] > 0.5).astype(np.float32), cmap='gray')
        axes[2, i].set_title(f"SSIM={vae_ssim[i]:.3f}")
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Autoencoder", fontsize=12)
    axes[2, 0].set_ylabel("VAE", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("comparison_reconstructions.png", dpi=150)
    plt.show()
    print("Saved: comparison_reconstructions.png")
    
    # Figure 2: Generated samples comparison
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle("Model Comparison: Real vs Autoencoder Generated vs VAE Generated", fontsize=14)
    
    for i in range(6):
        # Real
        axes[0, i].imshow(images[i*150], cmap='gray')
        axes[0, i].set_title(f"VF={real_vf[i*150]:.2f}")
        axes[0, i].axis('off')
        
        # Autoencoder generated
        axes[1, i].imshow((ae_generated[i] > 0.5).astype(np.float32), cmap='gray')
        axes[1, i].set_title(f"VF={ae_gen_vf[i]:.2f}")
        axes[1, i].axis('off')
        
        # VAE generated
        axes[2, i].imshow((vae_generated[i] > 0.5).astype(np.float32), cmap='gray')
        axes[2, i].set_title(f"VF={vae_gen_vf[i]:.2f}")
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel("Real\nSamples", fontsize=12)
    axes[1, 0].set_ylabel("Autoencoder\nGenerated", fontsize=12)
    axes[2, 0].set_ylabel("VAE\nGenerated", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("comparison_generated.png", dpi=150)
    plt.show()
    print("Saved: comparison_generated.png")
    
    # Figure 3: Metrics summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Quantitative Metrics Summary", fontsize=14)
    
    # SSIM comparison
    models = ['Autoencoder', 'VAE']
    ssim_means = [ae_ssim.mean(), vae_ssim.mean()]
    ssim_stds = [ae_ssim.std(), vae_ssim.std()]
    
    axes[0].bar(models, ssim_means, yerr=ssim_stds, capsize=5, color=['steelblue', 'coral'])
    axes[0].set_ylabel("SSIM Score")
    axes[0].set_title("Reconstruction Quality (SSIM)\nHigher is better")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(ssim_means):
        axes[0].text(i, v + ssim_stds[i] + 0.02, f'{v:.3f}', ha='center', fontsize=11)
    
    # Volume fraction comparison
    categories = ['Real', 'AE Recon', 'VAE Recon', 'AE Gen', 'VAE Gen']
    vf_means = [real_vf.mean(), ae_recon_vf.mean(), vae_recon_vf.mean(), 
                ae_gen_vf.mean(), vae_gen_vf.mean()]
    vf_stds = [real_vf.std(), ae_recon_vf.std(), vae_recon_vf.std(),
               ae_gen_vf.std(), vae_gen_vf.std()]
    colors = ['gray', 'steelblue', 'coral', 'lightblue', 'lightsalmon']
    
    axes[1].bar(categories, vf_means, yerr=vf_stds, capsize=5, color=colors)
    axes[1].set_ylabel("Volume Fraction")
    axes[1].set_title("Volume Fraction Comparison\nShould match Real")
    axes[1].axhline(y=real_vf.mean(), color='black', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=30)
    
    # MSE comparison
    mse_values = [ae_mse, vae_mse]
    axes[2].bar(models, mse_values, color=['steelblue', 'coral'])
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Reconstruction Error (MSE)\nLower is better")
    for i, v in enumerate(mse_values):
        axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("metrics_summary.png", dpi=150)
    plt.show()
    print("Saved: metrics_summary.png")
    
    # Figure 4: SSIM distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SSIM Score Distribution", fontsize=14)
    
    axes[0].hist(ae_ssim, bins=30, alpha=0.7, label=f'AE (μ={ae_ssim.mean():.3f})', color='steelblue')
    axes[0].hist(vae_ssim, bins=30, alpha=0.7, label=f'VAE (μ={vae_ssim.mean():.3f})', color='coral')
    axes[0].set_xlabel("SSIM Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SSIM Distribution")
    axes[0].legend()
    
    axes[1].hist(real_vf, bins=30, alpha=0.7, label=f'Real (μ={real_vf.mean():.3f})', color='gray')
    axes[1].hist(ae_gen_vf, bins=30, alpha=0.7, label=f'AE Gen (μ={ae_gen_vf.mean():.3f})', color='steelblue')
    axes[1].hist(vae_gen_vf, bins=30, alpha=0.7, label=f'VAE Gen (μ={vae_gen_vf.mean():.3f})', color='coral')
    axes[1].set_xlabel("Volume Fraction")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Volume Fraction Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("metrics_distribution.png", dpi=150)
    plt.show()
    print("Saved: metrics_distribution.png")
    
    # =============================================================================
    # PRINT SUMMARY TABLE
    # =============================================================================
    
    print("\n" + "="*60)
    print("SUMMARY TABLE (for your report)")
    print("="*60)
    print("""
┌─────────────────┬──────────────┬──────────────┐
│ Metric          │ Autoencoder  │ VAE          │
├─────────────────┼──────────────┼──────────────┤
│ SSIM (recon)    │ {:.4f}       │ {:.4f}       │
│ MSE (recon)     │ {:.6f}     │ {:.6f}     │
│ VF (generated)  │ {:.4f}       │ {:.4f}       │
│ VF (real)       │ {:.4f}       │ {:.4f}       │
└─────────────────┴──────────────┴──────────────┘
    """.format(
        ae_ssim.mean(), vae_ssim.mean(),
        ae_mse, vae_mse,
        ae_gen_vf.mean(), vae_gen_vf.mean(),
        real_vf.mean(), real_vf.mean()
    ))
    
    print("\n" + "="*60)
    print("COMPLETE! Files saved:")
    print("  - comparison_reconstructions.png")
    print("  - comparison_generated.png")
    print("  - metrics_summary.png")
    print("  - metrics_distribution.png")
    print("="*60)