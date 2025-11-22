# tarea5_autoencoder/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int):
    """Crea un kernel Gaussiano 2D para usar en SSIM."""
    coords = torch.arange(window_size).float() - window_size // 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()                      # normalizar

    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)  # producto externo
    g_2d = g_2d.unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
    g_2d = g_2d.repeat(channels, 1, 1, 1)        # (C,1,H,W) para groups=C
    return g_2d


class SSIMLoss(nn.Module):
    """
    Pérdida basada en SSIM.
    Devuelve 1 - SSIM, para que valores más bajos sean mejores.
    """
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        self.reduction = reduction

        # estos se actualizan en forward según el device/dtype del tensor
        self.register_buffer(
            "window",
            _gaussian_window(window_size, sigma, channel),
            persistent=False,
        )

        # asumimos imágenes en rango [0,1]
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim_map(self, x, y):
        # asegurar que el kernel esté en el mismo device/dtype
        if self.window.device != x.device or self.window.dtype != x.dtype:
            self.window = _gaussian_window(
                self.window_size, self.sigma, self.channel
            ).to(device=x.device, dtype=x.dtype)

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=self.channel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_y2
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_xy

        num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)

        ssim_map = num / (den + 1e-8)
        return ssim_map

    def forward(self, pred, target):
        """
        pred, target: tensores (N, C, H, W) en rango [0,1].
        """
        ssim_map = self._ssim_map(pred, target)

        if self.reduction == "mean":
            return 1.0 - ssim_map.mean()
        elif self.reduction == "sum":
            return 1.0 - ssim_map.sum()
        else:  # 'none'
            return 1.0 - ssim_map


class SSIML1Loss(nn.Module):
    """
    Combinación: alpha * SSIMLoss + (1 - alpha) * L1Loss.
    """
    def __init__(
        self,
        alpha: float = 0.84,
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIMLoss(
            window_size=window_size,
            channel=channel,
            sigma=sigma,
            reduction=reduction,
        )
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        loss_ssim = self.ssim(pred, target)
        loss_l1 = self.l1(pred, target)
        return self.alpha * loss_ssim + (1.0 - self.alpha) * loss_l1
