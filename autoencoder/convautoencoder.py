from typing import Any
from torch import optim, nn
import torch
import pytorch_lightning as pl


class ConvAutoEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, ),
            nn.ReLU(inplace=True),
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=1,),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                               stride=1, ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4,
                               stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> Any:
        z = self._encoder(x)
        x_hat = self._decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log("training_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, ),
            nn.ReLU(inplace=True),
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=1,),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                               stride=1, ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4,
                               stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> Any:
        z = self._encoder(x)
        x_hat = self._decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, y)

        self.log("training_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor):
        return x.view(*self.shape)


class VarationalAutoEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        self._sigma = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> Any:
        X = self._encoder(x)

        μ = self._mu(X)
        σ = self._sigma(X)
        ϵ = torch.randn_like(σ)
        
        z = ϵ * σ.pow(2) + μ

        x_hat = self._decoder(z.view(-1, 128, 1, 1))
        return x_hat, μ, σ

    def training_step(self, batch, batch_idx):
        x, = batch
        x_hat , μ, σ= self.forward(x)

        reconstruction_loss = nn.functional.mse_loss(x_hat, x)
        Dkl_loss = -0.5 * torch.sum(1+σ - μ.pow(2)-σ.exp())
        loss = (Dkl_loss + reconstruction_loss)/len(x_hat)
        self.log("training_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
