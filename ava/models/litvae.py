"""
PyTorch Lightning wrapper for the AVA VAE.

Implements a LightningModule that wraps ava.models.vae.VAE to preserve model
architecture and the original ELBO mathematics exactly. Provides clean
training/validation with checkpointing handled by the Lightning Trainer.

All public methods are kept short and type-annotated to satisfy strict typing
guidelines.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

try:
    from lightning import LightningModule  # type: ignore
except Exception as _e:  # pragma: no cover - import guard for environments w/o lightning
    raise RuntimeError("PyTorch Lightning is required. Please install `lightning>=2`.") from _e

from ava.models.vae import VAE, X_DIM


class LitVAE(LightningModule):
    """LightningModule that wraps the legacy VAE for training with Lightning."""

    def __init__(
        self,
        latent_dim: int = 32,
        model_precision: float = 10.0,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VAE(z_dim=latent_dim, lr=lr, model_precision=model_precision)
        self.beta_kl = beta_kl

    # --------- core loss (parity with legacy) ---------
    def _compute_terms(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (loss, recon_term, kl_term) matching legacy math."""
        mu, u, d = self.model.encode(x)
        dist = torch.distributions.LowRankMultivariateNormal(mu, u, d)
        z = dist.rsample()
        x_rec = self.model.decode(z)
        prior = -0.5 * (
            torch.sum(z.pow(2))
            + self.model.z_dim * torch.log(torch.tensor(2.0 * 3.141592653589793))
        )
        # recon term uses same constants as legacy forward
        pxz = (
            -0.5
            * X_DIM
            * (
                torch.log(torch.tensor(2.0 * 3.141592653589793))
                - torch.log(torch.tensor(self.model.model_precision))
            )
        )
        l2 = torch.sum((x.view(x.shape[0], -1) - x_rec).pow(2), dim=1)
        pxz = pxz - 0.5 * self.model.model_precision * torch.sum(l2)
        entropy = torch.sum(dist.entropy())
        elbo = prior + pxz + entropy
        kl = -(prior + entropy)
        loss = -(pxz - self.beta_kl * kl) if self.beta_kl != 1.0 else -elbo
        return loss, pxz.detach(), kl.detach()

    # --------- Lightning hooks ---------
    def forward(self, x: Tensor) -> Tensor:
        """Return reconstruction means with the legacy network."""
        _, _, rec = self.model.forward(x, return_latent_rec=True)
        return torch.from_numpy(rec)

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        loss, recon, kl = self._compute_terms(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon", -recon, on_step=False, on_epoch=True)
        self.log("train_kl", kl, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tensor, _: int) -> None:
        loss, recon, kl = self._compute_terms(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon", -recon, on_step=False, on_epoch=True)
        self.log("val_kl", kl, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.model.lr)

    # --------- utility ---------
    def get_example_reconstruction(self, x: Tensor) -> Tensor:
        """Return reconstruction for a batch as a tensor shaped [B, 1, H, W]."""
        _, _, rec = self.model.forward(x, return_latent_rec=True)
        rec_t = torch.from_numpy(rec).to(x.device)
        return rec_t.unsqueeze(1)
