import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class TrimOutput(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten()
        )

        self.z_mean = torch.nn.Linear(4096, 200)
        self.z_log_var = torch.nn.Linear(4096, 200)

        self.decoder = nn.Sequential(
            torch.nn.Linear(200, 4096),
            Reshape(-1, 64, 8, 8),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1),
            TrimOutput(),
            nn.Sigmoid()
        )

    def get_encodings(self, x):
        '''Returns latent embeddings'''
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x).to(DEVICE), self.z_log_var(x).to(DEVICE)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(DEVICE)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x).to(DEVICE), self.z_log_var(x).to(DEVICE)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

    def get_original_from_encoded(self, x):
        x = self.decoder(x)
        return x

