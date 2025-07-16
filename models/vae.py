# ------------------ PyTorch VAE for inference ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_PT(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder_dense_1 = nn.Linear(input_dim, 64)
        self.encoder_dense_2 = nn.Linear(64, 32)
        self.z_mean_layer = nn.Linear(32, latent_dim)
        self.z_log_var_layer = nn.Linear(32, latent_dim)
        # Decoder
        self.decoder_dense_1 = nn.Linear(latent_dim, 32)
        self.decoder_dense_2 = nn.Linear(32, 64)
        self.decoder_output = nn.Linear(64, input_dim)

    def encode(self, x):
        x = F.relu(self.encoder_dense_1(x))
        x = F.relu(self.encoder_dense_2(x))
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        d = F.relu(self.decoder_dense_1(z))
        d = F.relu(self.decoder_dense_2(d))
        return torch.sigmoid(self.decoder_output(d))

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z), z_mean, z_log_var

    def reconstruct(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z) 