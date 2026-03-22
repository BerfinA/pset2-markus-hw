"""
Problem Set 2 - Question 2: VAE at 128x128 Resolution
Higher resolution for sharper, more detailed results--> collapsed
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
# VAE FOR 128x128 IMAGES
# =============================================================================

class VAE_128(nn.Module):
    """VAE for 128x128 images - one extra conv layer."""
    
    def __init__(self, latent_dim=128):
        super(VAE_128, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 128 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 64 -> 128
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
        h = h.view(h.size(0), 512, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(model, train_loader, num_epochs=200, lr=1e-3, beta=0.1, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Slow beta annealing
        if epoch < 30:
            current_beta = 0.0
        else:
            current_beta = beta * min(1.0, (epoch - 30) / 70)
        
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon_x, batch_x, mu, logvar, current_beta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, β: {current_beta:.4f}")
    
    return losses


def load_data_128(npz_path):
    """Load data at 128x128 resolution."""
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
        pil_resized = pil_img.resize((128, 128), Image.LANCZOS)
        resized.append(np.array(pil_resized))
    
    images_resized = np.array(resized)
    images_norm = images_resized.astype(np.float32) / 255.0
    images_norm = (images_norm > 0.5).astype(np.float32)
    
    print(f"Resized to: {images_norm.shape}")
    return images_norm


def visualize_results_128(model, images, device, latent_codes=None):
    """Create all visualization figures for 128x128 model."""
    model.eval()
    
    # Get reconstructions
    with torch.no_grad():
        test_tensor = torch.FloatTensor(images[:6]).unsqueeze(1).to(device)
        recon, mu, _ = model(test_tensor)
        recon = recon.cpu().numpy()
        
        if latent_codes is None:
            all_tensor = torch.FloatTensor(images).unsqueeze(1).to(device)
            mu_all, _ = model.encode(all_tensor)
            latent_codes = mu_all.cpu().numpy()
    
    # Figure 1: Reconstructions
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("128x128 VAE: Original vs Reconstructed", fontsize=14)
    
    for i in range(6):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow((recon[i, 0] > 0.5).astype(np.float32), cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("128_reconstructions.png", dpi=150)
    plt.show()
    print("Saved: 128_reconstructions.png")
    
    # Figure 2: Generated samples
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    random_codes = np.random.randn(9, model.latent_dim) * std + mean
    
    with torch.no_grad():
        generated = model.decode(torch.FloatTensor(random_codes).to(device)).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("128x128 VAE: Generated Microstructures", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        ax.imshow((generated[idx, 0] > 0.5).astype(np.float32), cmap='gray')
        ax.set_title(f"Generated {idx+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("128_generated.png", dpi=150)
    plt.show()
    print("Saved: 128_generated.png")
    
    # Figure 3: Compare 64x64 vs 128x128 (side by side real samples)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("Resolution Comparison: Real Samples at 128x128", fontsize=14)
    
    for i in range(6):
        # Show real at 128x128
        axes[0, i].imshow(images[i*100], cmap='gray')
        axes[0, i].set_title(f"Real {i+1}")
        axes[0, i].axis('off')
        
        # Show generated at 128x128
        if i < 6:
            axes[1, i].imshow((generated[i, 0] > 0.5).astype(np.float32), cmap='gray')
            axes[1, i].set_title(f"Generated {i+1}")
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Real\n128x128", fontsize=12)
    axes[1, 0].set_ylabel("Generated\n128x128", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("128_comparison.png", dpi=150)
    plt.show()
    print("Saved: 128_comparison.png")
    
    # Figure 4: Interpolation
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
    
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5*num_steps, 3))
    fig.suptitle("128x128 VAE: Latent Space Interpolation", fontsize=14)
    
    for idx, ax in enumerate(axes):
        ax.imshow((interpolated[idx] > 0.5).astype(np.float32), cmap='gray')
        ax.set_title(f"α={alphas[idx]:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("128_interpolation.png", dpi=150)
    plt.show()
    print("Saved: 128_interpolation.png")
    
    return latent_codes


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    NPZ_PATH = "grf_data/MICRO2D_GRF_1k.npz"
    
    print("="*60)
    print("VAE at 128x128 RESOLUTION")
    print("="*60)
    
    # Load data at 128x128
    images = load_data_128(NPZ_PATH)
    
    # Create data loader
    train_tensor = torch.FloatTensor(images).unsqueeze(1)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for memory
    
    # Create model
    model = VAE_128(latent_dim=128)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\n>>> Training 128x128 VAE (200 epochs)...")
    print("This will take ~10 minutes...\n")
    
    losses = train_vae(model, train_loader, num_epochs=200, lr=1e-3, beta=0.1, device=device)
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("128x128 VAE Training Loss")
    plt.savefig("128_training_loss.png", dpi=150)
    plt.show()
    print("Saved: 128_training_loss.png")
    
    # Visualize results
    print("\n>>> Creating visualizations...")
    latent_codes = visualize_results_128(model, images, device)
    
    # Save model
    torch.save(model.state_dict(), "vae_128_model.pth")
    print("\nSaved model: vae_128_model.pth")
    
    print("\n" + "="*60)
    print("COMPLETE! 128x128 figures saved:")
    print("  - 128_training_loss.png")
    print("  - 128_reconstructions.png")
    print("  - 128_generated.png")
    print("  - 128_comparison.png")
    print("  - 128_interpolation.png")
    print("="*60)