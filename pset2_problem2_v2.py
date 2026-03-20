"""
Problem Set 2 - Question 2: Autoencoder for GRF Microstructures
IMPROVED VERSION - BCE loss, more epochs, better architecture
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
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
# IMPROVED Convolutional Autoencoder
# =============================================================================

class ImprovedAutoencoder(nn.Module):
    """
    Improved Autoencoder with:
    - Larger latent dimension (64)
    - Batch normalization for stable training
    - LeakyReLU to avoid dead neurons
    """
    
    def __init__(self, latent_dim=64):
        super(ImprovedAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8 -> 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_encode = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8 -> 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32 -> 64
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_autoencoder(model, train_loader, num_epochs=150, lr=1e-3, device='cpu'):
    """Train with BCE loss for binary images."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # BCE loss is better for binary images!
    criterion = nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            recon_x, _ = model(batch_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
    
    return losses


def load_and_prepare_data(npz_path, target_size=64):
    """Load and prepare the dataset."""
    print(f"\nLoading dataset from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    images = data['X']
    
    print(f"Original shape: {images.shape}")
    
    # Resize images
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
    
    # Normalize to [0, 1]
    images_norm = images_resized.astype(np.float32) / 255.0
    
    # Threshold to make binary (helps with BCE loss)
    images_norm = (images_norm > 0.5).astype(np.float32)
    
    print(f"Resized shape: {images_norm.shape}")
    print(f"Value range: [{images_norm.min()}, {images_norm.max()}]")
    
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
        recon, _ = model(test_tensor)
        recon = recon.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle("Part (c): Original vs Reconstructed", fontsize=14)
    
    for i in range(num_samples):
        axes[0, i].imshow(test_images[i], cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Threshold reconstruction to make it binary
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
        latent_codes = model.encode(tensor).cpu().numpy()
    
    print(f"\nLatent space shape: {latent_codes.shape}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Part (d): Latent Space Visualization", fontsize=14)
    
    dim_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for ax, (d1, d2) in zip(axes, dim_pairs):
        if d2 < latent_codes.shape[1]:
            ax.scatter(latent_codes[:, d1], latent_codes[:, d2], alpha=0.5, s=10)
            ax.set_xlabel(f"Dimension {d1}")
            ax.set_ylabel(f"Dimension {d2}")
            ax.set_title(f"Dim {d1} vs Dim {d2}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")
    
    return latent_codes


def generate_new_microstructures(model, latent_codes, device='cpu', num_samples=9,
                                  save_path="part_e_generated.png"):
    """Generate new microstructures from latent codes."""
    model.eval()
    
    # Sample from the learned latent distribution
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    
    # Sample closer to the mean for more realistic results
    random_codes = np.random.randn(num_samples, latent_codes.shape[1]) * std * 0.8 + mean
    
    with torch.no_grad():
        z = torch.FloatTensor(random_codes).to(device)
        generated = model.decode(z).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (e): Generated Microstructures (Random Latent Codes)", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        # Threshold to make binary
        gen_binary = (generated[idx, 0] > 0.5).astype(np.float32)
        ax.imshow(gen_binary, cmap='gray')
        ax.set_title(f"Generated {idx+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def interpolate_latent_space(model, images, device='cpu', save_path="part_e_interpolation.png"):
    """Interpolate between two microstructures in latent space."""
    model.eval()
    
    with torch.no_grad():
        # Encode two different images
        img1 = torch.FloatTensor(images[0:1]).unsqueeze(1).to(device)
        img2 = torch.FloatTensor(images[50:51]).unsqueeze(1).to(device)
        
        z1 = model.encode(img1)
        z2 = model.encode(img2)
        
        # Interpolate
        num_steps = 7
        alphas = np.linspace(0, 1, num_steps)
        
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img_interp = model.decode(z_interp)
            interpolated.append(img_interp.cpu().numpy()[0, 0])
    
    fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    NPZ_PATH = "grf_data/MICRO2D_GRF_1k.npz"
    
    print("="*60)
    print("PROBLEM 2: GRF Microstructure Autoencoder (IMPROVED)")
    print("="*60)
    
    # Load data
    images_norm, data = load_and_prepare_data(NPZ_PATH, target_size=64)
    
    # Part (a): Visualize samples
    print("\n>>> PART (a): Sample images...")
    visualize_samples(images_norm)
    
    # Create data loader
    train_tensor = torch.FloatTensor(images_norm).unsqueeze(1)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create improved model
    model = ImprovedAutoencoder(latent_dim=64)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Part (c): Train
    print("\n>>> PART (c): Training (150 epochs with BCE loss)...")
    print("This will take a few minutes...\n")
    losses = train_autoencoder(model, train_loader, num_epochs=150, lr=1e-3, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCE)")
    plt.title("Part (c): Training Loss (Improved)")
    plt.savefig("part_c_training_loss.png", dpi=150)
    plt.show()
    print("Saved: part_c_training_loss.png")
    
    # Visualize reconstructions
    print("\n>>> Reconstructions...")
    visualize_reconstructions(model, images_norm, device=device)
    
    # Part (d): Latent space
    print("\n>>> PART (d): Latent space visualization...")
    latent_codes = visualize_latent_space(model, images_norm, device=device)
    
    # Part (e): Generate new microstructures
    print("\n>>> PART (e): Generating new microstructures...")
    generate_new_microstructures(model, latent_codes, device=device)
    
    # Bonus: Interpolation
    print("\n>>> BONUS: Latent space interpolation...")
    interpolate_latent_space(model, images_norm, device=device)
    
    # Save model
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("\nSaved model: autoencoder_model.pth")
    
    print("\n" + "="*60)
    print("COMPLETE! Check the PNG files for your report.")
    print("="*60)

# #observations from pset 2 Key Observations for Your Report:
#Part (c) - Reconstructions: The model now reconstructs all 5 samples accurately, including the challenging diagonal stripe pattern (Sample 3). BCE loss + thresholding 
# produces crisp binary outputs.
#Part (d) - Latent Space: No strong correlations between dimensions (points are spread out, not clustered along diagonals). This is typical for a standard autoencoder.
#Part (e) - Generation: The generated images are now binary and show microstructure-like patterns, but they still have some artifacts (scattered pixels, less smooth boundaries 
# than real GRF). This is because random sampling doesn't guarantee valid latent codes.
#Interpolation (Bonus): The smooth transition from α=0.00 to α=1.00 shows the latent space is continuous and meaningful - you can morph between two real microstructures.
