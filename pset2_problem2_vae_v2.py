"""
Problem Set 2 - Question 2: VAE for GRF Microstructures
VERSION 2: Lower beta (0.1) + slower annealing for better generation
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch version: {torch.__version__}")

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =============================================================================
# VARIATIONAL AUTOENCODER
# =============================================================================

class VAE(nn.Module):
    def __init__(self, latent_dim=128):  # Increased latent dim
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder - deeper network
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
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
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
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
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    """VAE Loss with lower beta for better generation."""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(model, train_loader, num_epochs=250, lr=1e-3, beta=0.1, device='cpu'):
    """Train VAE with very slow beta annealing."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        # Very slow beta annealing: 0 for first 50 epochs, then slowly increase
        if epoch < 50:
            current_beta = 0.0  # Pure autoencoder first
        else:
            # Slowly ramp up beta from 0 to target over 100 epochs
            current_beta = beta * min(1.0, (epoch - 50) / 100)
        
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, recon, kl = vae_loss(recon_x, batch_x, mu, logvar, current_beta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
        
        n = len(train_loader.dataset)
        avg_loss = epoch_loss / n
        avg_recon = epoch_recon / n
        avg_kl = epoch_kl / n
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, β: {current_beta:.4f}")
    
    return losses, recon_losses, kl_losses


def load_and_prepare_data(npz_path, target_size=64):
    """Load and prepare the dataset."""
    print(f"\nLoading dataset from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    images = data['X']
    
    print(f"Original shape: {images.shape}")
    
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
    
    print(f"Resized shape: {images_norm.shape}")
    return images_norm, data


def visualize_samples(images, num_samples=9, save_path="part_a_sample_images.png"):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (a): Sample GRF Microstructures", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images) and idx < num_samples:
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f"Sample {idx+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def visualize_reconstructions(model, test_images, device='cpu', num_samples=5,
                              save_path="part_c_reconstructions.png"):
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_images[:num_samples]).unsqueeze(1).to(device)
        recon, _, _ = model(test_tensor)
        recon = recon.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle("Part (c): Original vs Reconstructed (VAE v2)", fontsize=14)
    
    for i in range(num_samples):
        axes[0, i].imshow(test_images[i], cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        recon_binary = (recon[i, 0] > 0.5).astype(np.float32)
        axes[1, i].imshow(recon_binary, cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def visualize_latent_space(model, images, device='cpu', save_path="part_d_latent_space.png"):
    model.eval()
    
    with torch.no_grad():
        tensor = torch.FloatTensor(images).unsqueeze(1).to(device)
        mu, logvar = model.encode(tensor)
        latent_codes = mu.cpu().numpy()
    
    print(f"\nLatent space shape: {latent_codes.shape}")
    print(f"Latent mean: {latent_codes.mean():.3f}, std: {latent_codes.std():.3f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Part (d): VAE Latent Space", fontsize=14)
    
    dim_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for ax, (d1, d2) in zip(axes, dim_pairs):
        ax.scatter(latent_codes[:, d1], latent_codes[:, d2], alpha=0.5, s=10)
        ax.set_xlabel(f"Dimension {d1}")
        ax.set_ylabel(f"Dimension {d2}")
        ax.set_title(f"Dim {d1} vs Dim {d2}")
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")
    
    return latent_codes


def generate_from_random(model, latent_codes, device='cpu', num_samples=9,
                         save_path="part_e_generated.png"):
    """Generate using learned distribution statistics."""
    model.eval()
    
    # Use statistics from the actual latent codes
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    
    # Sample from the learned distribution
    random_codes = np.random.randn(num_samples, model.latent_dim) * std + mean
    random_codes = torch.FloatTensor(random_codes).to(device)
    
    with torch.no_grad():
        generated = model.decode(random_codes).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (e): Generated Microstructures (VAE v2)", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        gen_binary = (generated[idx, 0] > 0.5).astype(np.float32)
        ax.imshow(gen_binary, cmap='gray')
        ax.set_title(f"Generated {idx+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def interpolate_latent_space(model, images, device='cpu', 
                             save_path="part_e_interpolation.png"):
    model.eval()
    
    with torch.no_grad():
        # Pick two visually different images
        img1 = torch.FloatTensor(images[0:1]).unsqueeze(1).to(device)
        img2 = torch.FloatTensor(images[50:51]).unsqueeze(1).to(device)
        
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        
        num_steps = 9
        alphas = np.linspace(0, 1, num_steps)
        
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp)
            interpolated.append(img_interp.cpu().numpy()[0, 0])
    
    fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2.5))
    fig.suptitle("Part (e): Latent Space Interpolation", fontsize=14)
    
    for idx, ax in enumerate(axes):
        img_binary = (interpolated[idx] > 0.5).astype(np.float32)
        ax.imshow(img_binary, cmap='gray')
        ax.set_title(f"α={alphas[idx]:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def compare_with_originals(model, images, latent_codes, device='cpu',
                           save_path="part_e_comparison.png"):
    """Compare generated samples with real samples side by side."""
    model.eval()
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("Part (e): Real vs Generated Microstructures", fontsize=14)
    
    # Row 1: Real samples
    for i in range(6):
        axes[0, i].imshow(images[i*100], cmap='gray')  # Spread out samples
        axes[0, i].set_title(f"Real {i+1}")
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel("Real\nSamples", fontsize=12)
    
    # Row 2: Generated samples from learned distribution
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    random_codes = np.random.randn(6, model.latent_dim) * std + mean
    random_codes = torch.FloatTensor(random_codes).to(device)
    
    with torch.no_grad():
        generated = model.decode(random_codes).cpu().numpy()
    
    for i in range(6):
        img = (generated[i, 0] > 0.5).astype(np.float32)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"Generated {i+1}")
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel("VAE\nGenerated", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    NPZ_PATH = "grf_data/MICRO2D_GRF_1k.npz"
    
    print("="*60)
    print("PROBLEM 2: VAE v2 (beta=0.1, slow annealing)")
    print("="*60)
    
    # Load data
    images_norm, data = load_and_prepare_data(NPZ_PATH, target_size=64)
    
    # Part (a)
    print("\n>>> PART (a): Sample images...")
    visualize_samples(images_norm)
    
    # Create data loader
    train_tensor = torch.FloatTensor(images_norm).unsqueeze(1)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create VAE with larger latent dim
    model = VAE(latent_dim=128)
    print(f"\nVAE Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train with lower beta and slow annealing
    print("\n>>> PART (c): Training VAE (250 epochs, beta=0.1, slow annealing)...")
    print("First 50 epochs: pure autoencoder (beta=0)")
    print("Epochs 50-150: slowly increase beta to 0.1")
    print("This will take ~5-10 minutes...\n")
    
    losses, recon_losses, kl_losses = train_vae(
        model, train_loader, num_epochs=250, lr=1e-3, beta=0.1, device=device
    )
    
    # Plot training loss
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Part (c): VAE Training Losses", fontsize=14)
    
    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    
    axes[1].plot(recon_losses)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("Reconstruction Loss (BCE)")
    
    axes[2].plot(kl_losses)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    
    plt.tight_layout()
    plt.savefig("part_c_training_loss.png", dpi=150)
    plt.show()
    print("Saved: part_c_training_loss.png")
    
    # Reconstructions
    print("\n>>> Reconstructions...")
    visualize_reconstructions(model, images_norm, device=device)
    
    # Part (d): Latent space
    print("\n>>> PART (d): Latent space visualization...")
    latent_codes = visualize_latent_space(model, images_norm, device=device)
    
    # Part (e): Generate
    print("\n>>> PART (e): Generating microstructures...")
    generate_from_random(model, latent_codes, device=device)
    
    # Interpolation
    print("\n>>> Interpolation...")
    interpolate_latent_space(model, images_norm, device=device)
    
    # Comparison
    print("\n>>> Comparison with real samples...")
    compare_with_originals(model, images_norm, latent_codes, device=device)
    
    # Save model
    torch.save(model.state_dict(), "vae_model_v2.pth")
    print("\nSaved model: vae_model_v2.pth")
    
    print("\n" + "="*60)
    print("COMPLETE! Check part_e_comparison.png for real vs generated.")
    print("="*60)