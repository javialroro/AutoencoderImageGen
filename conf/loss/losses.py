import torch
import torch.nn as nn
from pytorch_msssim import ssim


class SSIMLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5,
        **kwargs,
    ):
        super().__init__()
        self.reduction = reduction
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma

    def forward(self, x_rec, x):
        # x_rec y x en [0,1], shape (B, C, H, W)
        ssim_val = ssim(
            x_rec,
            x,
            data_range=1.0,
            size_average=True,  # igual que en el código de tu compa
        )
        loss = 1.0 - ssim_val
        return loss


class SSIML1Loss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.84,
        reduction: str = "mean",
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha

        # Reutilizamos la misma SSIMLoss de arriba (como tu compa)
        self.ssim = SSIMLoss(
            reduction=reduction,
            window_size=window_size,
            channel=channel,
            sigma=sigma,
        )
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, x_rec, x):
        ssim_loss = self.ssim(x_rec, x)          # = 1 - SSIM(x_rec, x)
        l1_loss = self.l1(x_rec, x)
        loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return loss


print("Funciones de pérdida SSIMLoss y SSIML1Loss definidas")
