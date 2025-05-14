import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

# ===== User Settings =====
time_limit_seconds = 10    # How long the animation should run
frame_interval_ms = 200    # Delay between frames (milliseconds)
save_path = "animated-gif/outputs/vae_training.gif"
# ==========================

# 1. Generate 2D data
X, _ = make_moons(n_samples=1000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

# 2. VAE Definition
class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, latent_dim)  # Mean
        self.fc22 = nn.Linear(16, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 3. Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# 4. Set up model, optimizer, data
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

# 5. Set up figure
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Original Data", "Latent Representation", "Reconstruction"]
scatters = [axs[i].scatter([], [], s=5) for i in range(3)]

for ax, title in zip(axs, titles):
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_title(title)

axs[1].set_xlim(-3, 3)  # Latent space x-axis
axs[1].set_ylim(-0.5, 0.5)  # Latent space y-axis (collapsed)

# 6. Animation update function
def update(frame):
    # Training step
    model.train()
    for batch, in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()

    # Evaluation step
    model.eval()
    with torch.no_grad():
        recon, mu, _ = model(X)

    # Update scatter plots
    scatters[0].set_offsets(X.numpy())
    scatters[1].set_offsets(torch.stack([mu[:, 0], torch.zeros_like(mu[:, 0])], dim=1).numpy())
    scatters[2].set_offsets(recon.numpy())
    
    return scatters

# 7. Calculate frames based on user settings
frames = (time_limit_seconds * 1000) // frame_interval_ms
print(f"Rendering {frames} frames (~{time_limit_seconds} seconds at {frame_interval_ms}ms per frame)")

# 8. Create animation
ani = animation.FuncAnimation(fig, update, frames=int(frames), interval=frame_interval_ms, blit=False)

# 9. Save animation
ani.save(save_path, writer='pillow', fps=1000//frame_interval_ms)
print(f"Saved animation as '{save_path}'!")

plt.show()

