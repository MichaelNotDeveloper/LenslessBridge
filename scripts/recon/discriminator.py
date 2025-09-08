import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicDiscriminator(nn.Module):
    """
    Basic PatchGAN-style discriminator for use as a GAN loss in reconstruction models.
    """

    def __init__(self, in_channels=1, ndf=64, n_layers=3, img_size=128):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        ndf : int
            Number of filters in the first conv layer.
        n_layers : int
            Number of convolutional layers.
        img_size : int
            Input image size (used for final flattening).
        """
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final conv layer
        layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Discriminator output (patch-level real/fake scores).
        """
        return self.model(x)

def discriminator_loss_fn(pred_real, pred_fake, real_target_generator, gen_target_generator):
    """
    Standard GAN loss for discriminator.

    Parameters
    ----------
    pred_real : torch.Tensor
        Discriminator predictions for real images.
    pred_fake : torch.Tensor
        Discriminator predictions for fake/generated images.

    Returns
    -------
    torch.Tensor
        Discriminator loss value.
    """
    loss_real = F.binary_cross_entropy_with_logits(pred_real, real_target_generator(pred_real.shape[0]))
    loss_fake = F.binary_cross_entropy_with_logits(pred_fake, gen_target_generator(pred_fake.shape[0]))
    return 0.5 * (loss_real + loss_fake)

def generator_loss_fn(pred_fake,  gen_target_generator):
    """
    Standard GAN loss for generator.

    Parameters
    ----------
    pred_fake : torch.Tensor
        Discriminator predictions for fake/generated images.

    Returns
    -------
    torch.Tensor
        Generator loss value.
    """
    return F.binary_cross_entropy_with_logits(pred_fake, gen_target_generator(pred_fake.shape[0]))
