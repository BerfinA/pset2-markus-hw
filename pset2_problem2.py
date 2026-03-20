"""
Problem Set 2 - Question 2: Autoencoder for GRF Microstructures
"""

import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available (Apple Silicon GPU): {torch.backends.mps.is_available()}")
except ImportError:
    print("PyTorch not installed")

# =============================================================================
# PART (a): Load and analyze the dataset
# =============================================================================

def load_npz_dataset(npz_path):
    """Load the GRF microstructure dataset from NPZ file."""
    print(f"\nLoading dataset from: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"Keys in NPZ file: {data.files}")
    
    # Get the images (stored in 'X')
    images = data['X']
    
    print(f"\nDataset shape: {images.shape}")
    print(f"Data type: {images.dtype}")
    print(f"Min value: {images.min():.4f}")
    print(f"Max value: {images.max():.4f}")
    
    return images, data


def visualize_samples(images, num_samples=9):
    """Part (a): Display sample microstructure images."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (a): Sample GRF Microstructures", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images) and idx < num_samples:
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f"Sample {idx+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("part_a_sample_images.png", dpi=150)
    plt.show()
    print("Saved: part_a_sample_images.png")


# =============================================================================
# PART (b): Choose image resolution
# =============================================================================

def analyze_resolution(images):
    """Part (b): Analyze and recommend resolution."""
    original_size = images.shape[1]  # Assuming square images
    
    print(f"\n" + "="*50)
    print("PART (b): Resolution Analysis")
    print("="*50)
    print(f"Original image size: {original_size}x{original_size}")
    
    # Test different resolutions
    resolutions = [32, 64, 128]
    
    print("\nMemory usage per batch of 32 images:")
    for res in resolutions:
        memory_mb = (32 * res * res * 4) / (1024 * 1024)  # float32
        print(f"  {res}x{res}: ~{memory_mb:.1f} MB")
    
    # Recommendation based on original size
    if original_size >= 256:
        recommended = 128
    elif original_size >= 128:
        recommended = 64
    else:
        recommended = 32
    
    print(f"\nRecommendation: {recommended}x{recommended}")
    print("This balances detail preservation with computational efficiency.")
    
    return recommended


def resize_images(images, target_size):
    """Resize images to target resolution."""
    from PIL import Image
    
    resized = []
    for img in images:
        # Normalize to 0-255 if needed
        if img.max() <= 1.0:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)
        
        pil_img = Image.fromarray(img_uint8)
        pil_resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
        resized.append(np.array(pil_resized))
    
    return np.array(resized)


# =============================================================================
# PART (c): Convolutional Autoencoder
# =============================================================================

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for microstructure images."""
    
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_encode = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
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


def train_autoencoder(model, train_loader, num_epochs=50, lr=1e-3, device='cpu'):
    """Train the autoencoder."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return losses


def visualize_reconstructions(model, test_images, device='cpu', num_samples=5):
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
        
        axes[1, i].imshow(recon[i, 0], cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("part_c_reconstructions.png", dpi=150)
    plt.show()
    print("Saved: part_c_reconstructions.png")


# =============================================================================
# PART (d): Explore latent space
# =============================================================================

def visualize_latent_space(model, images, device='cpu'):
    """Part (d): Visualize 2D projections of latent space."""
    model.eval()
    
    with torch.no_grad():
        tensor = torch.FloatTensor(images).unsqueeze(1).to(device)
        latent_codes = model.encode(tensor).cpu().numpy()
    
    print(f"\nLatent space shape: {latent_codes.shape}")
    
    # Plot different dimension pairs
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
    plt.savefig("part_d_latent_space.png", dpi=150)
    plt.show()
    print("Saved: part_d_latent_space.png")
    
    # Check correlations
    print("\nCorrelation matrix (first 8 dimensions):")
    corr = np.corrcoef(latent_codes[:, :8].T)
    print(np.round(corr, 2))
    
    return latent_codes


# =============================================================================
# PART (e): Generate new microstructures
# =============================================================================

def generate_new_microstructures(model, latent_codes, device='cpu', num_samples=9):
    """Part (e): Generate new microstructures from latent codes."""
    model.eval()
    
    # Method 1: Random sampling from latent space statistics
    mean = latent_codes.mean(axis=0)
    std = latent_codes.std(axis=0)
    
    random_codes = np.random.randn(num_samples, latent_codes.shape[1]) * std + mean
    
    with torch.no_grad():
        z = torch.FloatTensor(random_codes).to(device)
        generated = model.decode(z).cpu().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Part (e): Generated Microstructures (Random Latent Codes)", fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        ax.imshow(generated[idx, 0], cmap='gray')
        ax.set_title(f"Generated {idx+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("part_e_generated.png", dpi=150)
    plt.show()
    print("Saved: part_e_generated.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Path to the NPZ file
    NPZ_PATH = "grf_data/MICRO2D_GRF_1k.npz"
    
    print("="*60)
    print("PROBLEM 2: GRF Microstructure Autoencoder")
    print("="*60)
    
    # Part (a): Load and visualize
    print("\n>>> PART (a): Loading and visualizing dataset...")
    images, data = load_npz_dataset(NPZ_PATH)
    visualize_samples(images)
    
    # Part (b): Choose resolution
    print("\n>>> PART (b): Analyzing resolution...")
    target_size = analyze_resolution(images)
    
    # Resize if needed
    if images.shape[1] != 64:
        print(f"\nResizing images to 64x64...")
        images_resized = resize_images(images, 64)
    else:
        images_resized = images
    
    # Normalize to [0, 1]
    images_norm = images_resized.astype(np.float32) / 255.0
    print(f"Normalized images shape: {images_norm.shape}")
    
    # Part (c): Train autoencoder
    print("\n>>> PART (c): Training Convolutional Autoencoder...")
    
    # Setup device (use MPS on Apple Silicon if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create data loader
    train_tensor = torch.FloatTensor(images_norm).unsqueeze(1)  # Add channel dimension
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create and train model
    model = ConvAutoencoder(latent_dim=32)
    print(f"\nModel architecture:\n{model}")
    
    print("\nTraining...")
    losses = train_autoencoder(model, train_loader, num_epochs=50, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Part (c): Training Loss")
    plt.savefig("part_c_training_loss.png", dpi=150)
    plt.show()
    
    # Visualize reconstructions
    visualize_reconstructions(model, images_norm, device=device)
    
    # Part (d): Explore latent space
    print("\n>>> PART (d): Exploring latent space...")
    latent_codes = visualize_latent_space(model, images_norm, device=device)
    
    # Part (e): Generate new microstructures
    print("\n>>> PART (e): Generating new microstructures...")
    generate_new_microstructures(model, latent_codes, device=device)
    
    print("\n" + "="*60)
    print("COMPLETE! Check the saved PNG files for your report.")
    print("="*60)