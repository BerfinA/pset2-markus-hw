"""
Problem Set 2 - Question 2: VAE for GRF Microstructures
Variational Autoencoder - better for generation!
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
    """
    Variational Autoencoder:
    - Learns a structured latent space (Gaussian)
    - Random sampling produces realistic outputs
    - Uses reparameterization trick for backprop
    """
    
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
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
        
        # Latent space - mu and log_var (for reparameterization)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    beta controls the weight of KL divergence:
    - beta=1: standard VAE
    - beta<1: better reconstruction, less structured latent space
    - beta>1: more structured latent space, blurrier reconstruction
    """
    # Reconstruction loss (BCE for binary images)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: how far is the latent distribution from N(0,1)?
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(model, train_loader, num_epochs=200, lr=1e-3, beta=0.5, device='cpu'):
    """Train VAE with annealing beta schedule."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        # Beta annealing: start low, increase over time
        current_beta = min(beta, beta * (epoch / 50))
        
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
        
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, β: {current_beta:.3f}")
    
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
    """Display sample images."""
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
    """Visualize original vs reconstructed images."""
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_images[:num_samples]).unsqueeze(1).to(device)
        recon, _, _ = model(test_tensor)
        recon = recon.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle("Part (c): Original vs Reconstructed (VAE)", fontsize=14)
    
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
    """Visualize 2D projections of latent space."""
    model.eval()
    
    with torch.no_grad():
        tensor = torch.FloatTensor(images).unsqueeze(1).to(device)
        mu, logvar = model.encode(tensor)
        latent_codes = mu.cpu().numpy()  # Use mean for visualization
    
    print(f"\nLatent space shape: {latent_codes.shape}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Part (d): VAE Latent Space (should be more Gaussian!)", fontsize=14)
    
    dim_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for ax, (d1, d2) in zip(axes, dim_pairs):
        ax.scatter(latent_codes[:, d1], latent_codes[:, d2], alpha=0.5, s=10)
        ax.set_xlabel(f"Dimension {d1}")
        ax.set_ylabel(f"Dimension {d2}")
        ax.set_title(f"Dim {d1} vs Dim {d2}")
        
        # Draw unit circle for reference (latent should be close to N(0,1))
        circle = plt.Circle((0, 0), 2, fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")
    
    return latent_codes


def generate_from_random(model, device='cpu', num_samples=9,
                         save_path="part_e_generated.png"):
    """Generate microstructures from random N(0,1) samples - VAE should do this well!"""
    model.eval()
    
    # Sample from standard normal - this is the key advantage of VAE!
    random_codes = torch.randn(num_samples, model.latent_dim).to(device)
    
    with torch.no_grad():
        generated = model.decode(random_codes).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (e): Generated from Random N(0,1) - VAE", fontsize=14)
    
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
    """Interpolate between two microstructures."""
    model.eval()
    
    with torch.no_grad():
        img1 = torch.FloatTensor(images[0:1]).unsqueeze(1).to(device)
        img2 = torch.FloatTensor(images[50:51]).unsqueeze(1).to(device)
        
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        
        num_steps = 7
        alphas = np.linspace(0, 1, num_steps)
        
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp)
            interpolated.append(img_interp.cpu().numpy()[0, 0])
    
    fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
    fig.suptitle("Part (e): Latent Space Interpolation (VAE)", fontsize=14)
    
    for idx, ax in enumerate(axes):
        img_binary = (interpolated[idx] > 0.5).astype(np.float32)
        ax.imshow(img_binary, cmap='gray')
        ax.set_title(f"α={alphas[idx]:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def compare_generation_methods(model, latent_codes, device='cpu',
                               save_path="part_e_comparison.png"):
    """Compare random N(0,1) sampling vs sampling from learned distribution."""
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Part (e): VAE Generation Comparison", fontsize=14)
    
    # Row 1: Sample from N(0,1) - pure random
    random_z = torch.randn(5, model.latent_dim).to(device)
    with torch.no_grad():
        random_gen = model.decode(random_z).cpu().numpy()
    
    for i in range(5):
        img = (random_gen[i, 0] > 0.5).astype(np.float32)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Random N(0,1)")
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel("Random\nSampling", fontsize=12)
    
    # Row 2: Sample from learned distribution
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    learned_z = torch.FloatTensor(np.random.randn(5, model.latent_dim) * std + mean).to(device)
    
    with torch.no_grad():
        learned_gen = model.decode(learned_z).cpu().numpy()
    
    for i in range(5):
        img = (learned_gen[i, 0] > 0.5).astype(np.float32)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"Learned Dist")
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel("Learned\nDistribution", fontsize=12)
    
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
    print("PROBLEM 2: VAE for GRF Microstructures")
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
    
    # Create VAE
    model = VAE(latent_dim=64)
    print(f"\nVAE Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    print("\n>>> PART (c): Training VAE (200 epochs)...")
    print("This will take a few minutes...\n")
    losses, recon_losses, kl_losses = train_vae(
        model, train_loader, num_epochs=200, lr=1e-3, beta=0.5, device=device
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
    print("\n>>> PART (e): Generating from random N(0,1)...")
    generate_from_random(model, device=device)
    
    # Interpolation
    print("\n>>> Interpolation...")
    interpolate_latent_space(model, images_norm, device=device)
    
    # Comparison
    print("\n>>> Comparison of generation methods...")
    compare_generation_methods(model, latent_codes, device=device)
    
    # Save model
    torch.save(model.state_dict(), "vae_model.pth")
    print("\nSaved model: vae_model.pth")
    
    print("\n" + "="*60)
    print("COMPLETE! VAE should generate better microstructures.")
    print("="*60)
    
    
   # Results-------- 
'''
   Good news:

Latent space is now Gaussian! Look at Part (d) - the points are centered around (0,0) and mostly within the red circle (radius 2). This is exactly what VAE should do.
Reconstructions are still sharp - Sample 3 (stripes) still works well
Interpolation is smooth - transitions nicely between microstructures

Problem:

Generated images are too sparse! The Random N(0,1) samples produce mostly black with scattered white blobs - not realistic GRF microstructures
The "Learned Dist" row in the comparison looks better but still not great

Why this happened: The KL divergence collapsed too fast (see the KL plot - it drops to near zero immediately). This means the VAE is prioritizing making the latent space Gaussian over learning good reconstructions.
'''