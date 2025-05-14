import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

# 1. Generate 2D data (moons just for fun)
X, _ = make_moons(n_samples=1000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

# 2. Create a simple VAE
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

# 4. Train the model
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

for epoch in range(50):
    total_loss = 0
    for batch, in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.2f}")

# 5. Visualize: original data, latent space, reconstructions
with torch.no_grad():
    recon, mu, _ = model(X)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', s=5)
    plt.title("Original Data")

    plt.subplot(1, 3, 2)
    plt.scatter(mu[:, 0], torch.zeros_like(mu[:, 0]), c='red', s=5)
    plt.title("Latent Representation (1D)")

    plt.subplot(1, 3, 3)
    plt.scatter(recon[:, 0], recon[:, 1], c='green', s=5)
    plt.title("Reconstructed Data")

    plt.tight_layout()
    plt.show()

