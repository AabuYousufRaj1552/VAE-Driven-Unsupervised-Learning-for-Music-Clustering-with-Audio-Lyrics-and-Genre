"""
VAE Models for Hybrid Music Clustering
Includes: MLP-VAE, ConvVAE, CVAE, Beta-VAE, and Multi-modal VAE
"""

import numpy as np
import torch
import torch.nn as nn


class MLPVAE(nn.Module):
    """Basic MLP-based Variational Autoencoder"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class ConvVAE(nn.Module):
    """Convolutional VAE for 2D audio features (MFCC/Mel-spectrogram)"""
    
    def __init__(self, in_channels=1, latent_dim=32, F=40, T=258):
        super().__init__()
        self.F = F
        self.T = T

        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, F, T)
            h = self.enc_conv(dummy)
            self.conv_out_shape = h.shape[1:]
            self.flat_dim = int(np.prod(self.conv_out_shape))

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def crop_or_pad(self, x, F, T):
        n, c, f, t = x.shape
        x = x[:, :, :min(f, F), :min(t, T)]
        f2, t2 = x.shape[2], x.shape[3]
        if f2 < F or t2 < T:
            pad_f = F - f2
            pad_t = T - t2
            x = nn.functional.pad(x, (0, pad_t, 0, pad_f))
        return x

    def forward(self, x):
        h = self.enc_conv(x)
        h_flat = h.view(x.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_dec(z).view(x.size(0), *self.conv_out_shape)
        x_hat = self.dec_deconv(h_dec)
        x_hat = self.crop_or_pad(x_hat, self.F, self.T)
        return x_hat, mu, logvar, z


class MultiModalVAE(nn.Module):
    """Multi-modal VAE for Audio + Lyrics + Genre"""
    
    def __init__(self, F, T, lyrics_dim, genre_dim, latent_dim=32, use_cvae=False):
        super().__init__()
        self.F, self.T = F, T
        self.lyrics_dim = lyrics_dim
        self.genre_dim = genre_dim
        self.latent_dim = latent_dim
        self.use_cvae = use_cvae

        # Audio encoder
        self.a_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, F, T)
            h = self.a_enc(dummy)
            self.a_shape = h.shape[1:]
            self.a_flat = int(np.prod(self.a_shape))

        # Lyrics encoder
        self.l_enc = nn.Sequential(
            nn.Linear(lyrics_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # Genre encoder
        self.g_enc = nn.Sequential(
            nn.Linear(genre_dim, 64), nn.ReLU(),
        )

        enc_in = self.a_flat + 128 + 64 + (genre_dim if use_cvae else 0)
        self.enc_fc = nn.Sequential(
            nn.Linear(enc_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        dec_z_in = latent_dim + (genre_dim if use_cvae else 0)

        # Audio decoder
        self.a_dec_fc = nn.Linear(dec_z_in, self.a_flat)
        self.a_dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

        # Lyrics decoder
        self.l_dec = nn.Sequential(
            nn.Linear(dec_z_in, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, lyrics_dim),
        )

        # Genre decoder
        self.g_dec = nn.Sequential(
            nn.Linear(dec_z_in, 64), nn.ReLU(),
            nn.Linear(64, genre_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def crop_or_pad(self, x):
        n, c, f, t = x.shape
        x = x[:, :, :min(f, self.F), :min(t, self.T)]
        f2, t2 = x.shape[2], x.shape[3]
        if f2 < self.F or t2 < self.T:
            x = nn.functional.pad(x, (0, self.T - t2, 0, self.F - f2))
        return x

    def encode(self, xa, xl, xg):
        ha = self.a_enc(xa).view(xa.size(0), -1)
        hl = self.l_enc(xl)
        hg = self.g_enc(xg)
        if self.use_cvae:
            h = torch.cat([ha, hl, hg, xg], dim=1)
        else:
            h = torch.cat([ha, hl, hg], dim=1)
        h = self.enc_fc(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def decode(self, z, xg):
        if self.use_cvae:
            zc = torch.cat([z, xg], dim=1)
        else:
            zc = z

        ha = self.a_dec_fc(zc).view(z.size(0), *self.a_shape)
        xa_hat = self.a_dec(ha)
        xa_hat = self.crop_or_pad(xa_hat)
        xl_hat = self.l_dec(zc)
        xg_logits = self.g_dec(zc)

        return xa_hat, xl_hat, xg_logits

    def forward(self, xa, xl, xg):
        mu, logvar = self.encode(xa, xl, xg)
        z = self.reparameterize(mu, logvar)
        xa_hat, xl_hat, xg_logits = self.decode(z, xg)
        return xa_hat, xl_hat, xg_logits, mu, logvar, z


class SimpleAutoencoder(nn.Module):
    """Standard Autoencoder (non-variational) for baseline comparison"""
    
    def __init__(self, in_channels=1, latent_dim=64, F=40, T=258):
        super().__init__()
        self.F, self.T = F, T
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, F, T)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]
            self.flat_dim = int(np.prod(self.enc_shape))
        
        self.fc_latent = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h_flat = h.view(x.size(0), -1)
        z = self.fc_latent(h_flat)
        h_dec = self.fc_decode(z).view(x.size(0), *self.enc_shape)
        x_hat = self.decoder(h_dec)
        return x_hat, z


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    """Standard VAE loss: Reconstruction + KL divergence"""
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()


def multimodal_vae_loss(xa, xa_hat, xl, xl_hat, xg, xg_logits, mu, logvar, 
                        beta=1.0, w_audio=1.0, w_lyrics=1.0, w_genre=0.5):
    """Multi-modal VAE loss"""
    loss_a = nn.functional.mse_loss(xa_hat, xa, reduction="mean")
    loss_l = nn.functional.mse_loss(xl_hat, xl, reduction="mean")
    bce = nn.BCEWithLogitsLoss()
    loss_g = bce(xg_logits, xg)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = w_audio * loss_a + w_lyrics * loss_l + w_genre * loss_g + beta * kl
    return total, loss_a.detach(), loss_l.detach(), loss_g.detach(), kl.detach()
