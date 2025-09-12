# simple_discriminator.py
from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def dcgan_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvBlock(nn.Module):
    """
    Простой блок: Conv2d(stride=2) -> (опц. нормализация) -> LeakyReLU.
    Для дискриминатора нормализацию чаще не используют; по умолчанию отключена.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_norm: bool = False,          # InstanceNorm2d/BatchNorm2d в дискриминаторе обычно НЕ нужны
        norm_type: str = "bn",           # "bn" или "in"
        negative_slope: float = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=(not use_norm)),
        ]
        if use_norm:
            if norm_type == "in":
                layers.append(nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False))
            else:
                layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineDiscriminator(nn.Module):
    """
    Простой дискриминатор в стиле DCGAN.

    Параметры:
      in_channels:   каналы входа (напр. 3 для RGB)
      img_size:      входное разрешение (int или (H,W)); даунсэмпл в 2 раза на слой
      num_layers:    количество свёрточных слоёв со stride=2
      base_channels: число каналов на первом слое
      max_channels:  верхний предел каналов
      use_norm:      добавлять ли нормализацию после conv (по умолчанию False)
      norm_type:     "bn" или "in", если use_norm=True
      patch_output:  True -> вернуть PatchGAN-карту; False -> один скаляр на изображение
    """
    def __init__(
        self,
        in_channels: int = 3,
        img_size: Union[int, Tuple[int, int]] = 256,
        num_layers: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        use_norm: bool = False,
        norm_type: str = "bn",
        patch_output: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
        assert h > 0 and w > 0

        layers = []
        in_ch = in_channels
        ch = base_channels

        # Первый слой без нормализации (классика DCGAN)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(negative_slope, inplace=True),
            )
        )
        in_ch = ch

        # Остальные слои
        for _ in range(1, num_layers):
            ch = min(ch * 2, max_channels)
            layers.append(ConvBlock(in_ch, ch, use_norm=use_norm, norm_type=norm_type, negative_slope=negative_slope))
            in_ch = ch

        self.features = nn.Sequential(*layers)
        self.patch_output = patch_output

        # Головы
        if self.patch_output:
            # 1×1 conv выдаёт карту реалистичности
            self.head = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            # Глобальная оценка: усредним spatial и линейный слой
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_ch, 1)

        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        if self.patch_output:
            return self.head(h)  # (B,1,H',W')
        else:
            h = self.pool(h).flatten(1)    # (B,C)
            return self.fc(h)              # (B,1)


# ===== Примеры использования =====
if __name__ == "__main__":
    # Пример PatchGAN (карта)
    D_patch = SimpleDiscriminator(
        in_channels=3,
        img_size=256,
        num_layers=4,      # 4 раза поделим размер пополам: 256->128->64->32->16, выходная карта ~16x16
        base_channels=64,
        patch_output=True, # вернуть карту
        use_norm=False,    # по умолчанию без нормализации — стабильнее для D
    )
    x = torch.randn(8, 3, 256, 256)
    y = D_patch(x)
    print("Patch output:", y.shape)  # ожидаемо: (8, 1, ~16, ~16)

    # Пример глобального скора (скаляр)
    D_global = SimpleDiscriminator(
        in_channels=1,
        img_size=(128, 128),
        num_layers=3,       # 128->64->32->16
        base_channels=32,
        patch_output=False, # вернуть скаляр
        use_norm=False,
    )
    xg = torch.randn(8, 1, 128, 128)
    yg = D_global(xg)
    print("Global score:", yg.shape)  # (8, 1)
