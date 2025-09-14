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
        patch_output: bool = False,
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

    def generator_loss_fn(self, generated_images: torch.Tensor, target_generator=None) -> torch.Tensor:
        """
        Вычисляет лосс для генератора.
        
        Parameters
        ----------
        fake_logits : torch.Tensor
            Логиты дискриминатора для сгенерированных изображений.
        target_generator : callable, optional
            Функция для генерации целевых значений. Если None, используется self.gen_target_generator.
            
        Returns
        -------
        torch.Tensor
            Лосс генератора.
        """
        fake_logits = self.forward(generated_images)
        
        if target_generator is None:
            target_generator = self.gen_target_generator
        
        # Генерируем целевые значения
        batch_size = fake_logits.shape[0]
        if isinstance(target_generator, str):
            if target_generator == "Rand":
                print("Random generator used")
                # Случайные значения в диапазоне [0.0, 0.3]
                targets = torch.rand(batch_size, device=fake_logits.device) * 0.3
            elif target_generator == "Determ":
                # Детерминированные значения (например, 0.1)
                targets = torch.full((batch_size,), 0.1, device=fake_logits.device)
            else:
                raise ValueError(f"Unknown target_generator type: {target_generator}")
        else:
            # Если это функция, вызываем её
            targets = target_generator(batch_size)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(fake_logits.device)
            else:
                targets = torch.tensor(targets, device=fake_logits.device)
        
        # Вычисляем MSE лосс
        print(fake_logits.shape, targets.shape)
        loss = torch.log(torch.sum(torch.sigmoid(fake_logits.squeeze())-targets, axis = 0))
        return loss

    def discriminator_loss_fn(self, generated_images: torch.Tensor, real_images: torch.Tensor, 
                            real_target_generator=None, fake_target_generator=None) -> torch.Tensor:
        """
        Вычисляет лосс для дискриминатора.
        
        Parameters
        ----------
        real_logits : torch.Tensor
            Логиты дискриминатора для реальных изображений.
        fake_logits : torch.Tensor
            Логиты дискриминатора для сгенерированных изображений.
        real_target_generator : callable or str, optional
            Функция для генерации целевых значений для реальных изображений.
        fake_target_generator : callable or str, optional
            Функция для генерации целевых значений для сгенерированных изображений.
            
        Returns
        -------
        torch.Tensor
            Лосс дискриминатора.
        """
        if real_target_generator is None:
            real_target_generator = self.real_target_generator
        if fake_target_generator is None:
            fake_target_generator = self.gen_target_generator
        
        real_logits = self.forward(real_images)
        fake_logits = self.forward(generated_images)
        
        batch_size = real_logits.shape[0]
        
        # Генерируем целевые значения для реальных изображений
        if isinstance(real_target_generator, str):
            if real_target_generator == "Rand":
                # Случайные значения в диапазоне [0.7, 1.2]
                real_targets = torch.rand(batch_size, device=real_logits.device) * 0.5 + 0.7
            elif real_target_generator == "Determ":
                # Детерминированные значения (например, 0.9)
                real_targets = torch.full((batch_size,), 0.9, device=real_logits.device)
            else:
                raise ValueError(f"Unknown real_target_generator type: {real_target_generator}")
        else:
            real_targets = real_target_generator(batch_size)
            if isinstance(real_targets, torch.Tensor):
                real_targets = real_targets.to(real_logits.device)
            else:
                real_targets = torch.tensor(real_targets, device=real_logits.device)
        
        # Генерируем целевые значения для сгенерированных изображений
        if isinstance(fake_target_generator, str):
            if fake_target_generator == "Rand":
                # Случайные значения в диапазоне [0.0, 0.3]
                fake_targets = torch.rand(batch_size, device=fake_logits.device) * 0.3
            elif fake_target_generator == "Determ":
                # Детерминированные значения (например, 0.1)
                fake_targets = torch.full((batch_size,), 0.1, device=fake_logits.device)
            else:
                raise ValueError(f"Unknown fake_target_generator type: {fake_target_generator}")
        else:
            fake_targets = fake_target_generator(batch_size)
            if isinstance(fake_targets, torch.Tensor):
                fake_targets = fake_targets.to(fake_logits.device)
            else:
                fake_targets = torch.tensor(fake_targets, device=fake_logits.device)
        
        # Вычисляем MSE лоссы
        real_loss = torch.log(torch.sum(torch.sigmoid(real_logits.squeeze())-real_targets, axis = 0))
        fake_loss = torch.log(torch.sum(torch.sigmoid(fake_logits.squeeze())-fake_targets, axis = 0))
        
        # Общий лосс дискриминатора
        total_loss = 0.5 * (real_loss + fake_loss)
       
        # Лосс для прогрева дискриминатора
        real_loss_abs = torch.nn.L1Loss()(real_logits.squeeze(), real_targets)
        fake_loss_abs = torch.nn.L1Loss()(fake_logits.squeeze(), fake_targets)
        delta_score = torch.abs(real_loss_abs - fake_loss_abs)

        return total_loss, delta_score

    def get_target_generator(self, target_type: str, **kwargs):
        """
        Создает функцию генерации целевых значений в зависимости от типа.
        
        Parameters
        ----------
        target_type : str
            Тип генерации: "Rand" для случайных значений, "Determ" для детерминированных.
        **kwargs
            Дополнительные параметры для генерации.
            
        Returns
        -------
        callable
            Функция генерации целевых значений.
        """
        if target_type == "Rand":
            min_val = kwargs.get('min_val', 0.0)
            max_val = kwargs.get('max_val', 0.3)
            def rand_generator(batch_size):
                return torch.rand(batch_size) * (max_val - min_val) + min_val
            return rand_generator
        elif target_type == "Determ":
            value = kwargs.get('value', 0.1)
            def determ_generator(batch_size):
                return torch.full((batch_size,), value)
            return determ_generator
        else:
            raise ValueError(f"Unknown target_type: {target_type}. Use 'Rand' or 'Determ'")
