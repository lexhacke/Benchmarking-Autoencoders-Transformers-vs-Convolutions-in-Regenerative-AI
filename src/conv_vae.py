import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attention import SpatialAttention
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class ResBlock(nn.Module):
    """
    ResNet-Style Residual Block Module with Group Normalization and Conv2d layers.
    - Applies automatic channel reshaping if input and output channels differ.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        assert in_c % 32 == 0, f"Channels must be divisible by Groups (32) recieved: {in_c}"
        self.in_c = in_c
        self.out_c = out_c
        self.reshape = False
        if in_c != out_c:
            self.reshape = True
            self.conv_reshape = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_c, f"expected {self.in_c} channels, got {C}"
        if self.reshape:
            x = self.conv_reshape(x)
        res = x
        x = self.norm1(x)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)
        x = x + res
        return x

class Encoder(nn.Module):
    """
    Convolutional Encoder Module for 2D images.
    - Takes an input image of form B, 3, H, W and encodes it into a latent representation of shape B, D, hw
    - D = z_channels, hw = HW/2^(2n) where n is len(filters)
    - Compression ratio is ultimately D/(3 * 2^(2n)), so with z_channels=4 and n=4, we get 4/(3*256) = 1/192
    """
    def __init__(self, z_channels, hw, filters, attn_resolutions, depth):
        super().__init__()
        self.z_channels = z_channels
        self.out_shape = (z_channels, hw // 2**len(filters), hw // 2**len(filters))
        self.conv_out = nn.Conv2d(filters[-1], 2*z_channels, kernel_size=3, stride=1, padding=1)

        self.prep = nn.Conv2d(3, filters[0], kernel_size=3, stride=2, padding=1)
        self.down = nn.ModuleList()

        current_res = hw
        for i in range(len(filters)-1):
            current_res = current_res // 2
            block = nn.ModuleList([ResBlock(filters[i], filters[i+1])])
            for _ in range(depth-1):
                block.append(ResBlock(filters[i+1], filters[i+1]))
            if current_res in attn_resolutions:
                block.append(SpatialAttention(filters[i+1]))
            block.append(nn.Conv2d(filters[i+1], filters[i+1], kernel_size=3, stride=2, padding=1))
            self.down.append(block)

        self.mid = nn.Sequential(ResBlock(filters[-1], filters[-1]),
                                SpatialAttention(filters[-1]),
                                ResBlock(filters[-1], filters[-1]))

        self.norm = nn.GroupNorm(num_groups=32, num_channels=filters[-1], eps=1e-6, affine=True)

    def forward(self, x):
        x = self.prep(x)
        for block in self.down:
            for layer in block:
                x = layer(x)
        x = self.mid(x)
        x = self.norm(x)
        x = self.conv_out(x)
        x = rearrange(x, "B z H W -> B z (H W)")
        dist = DiagonalGaussianDistribution(x)
        return dist

class Decoder(nn.Module):
    """
    Convolutional Decoder Module for 2D latents.
    - Takes an input latent of form B, D, hw^2 and decodes it into B, 3, H, W image(s).
    """
    def __init__(self, z_channels, latent_hw, filters, attn_resolutions, depth):
        super().__init__()
        self.latent_hw = latent_hw
        self.z_channels = z_channels
        self.out_shape = (3, latent_hw*2**len(filters), latent_hw*2**len(filters))
        self.conv_in = nn.Conv2d(z_channels, filters[0], kernel_size=3, stride=1, padding=1)
        self.mid = nn.Sequential(ResBlock(filters[0], filters[0]),
                                 SpatialAttention(filters[0]),
                                 SpatialAttention(filters[0]),
                                 ResBlock(filters[0], filters[0]))

        self.up = nn.ModuleList()
        current_res = latent_hw
        for i in range(len(filters)-1):
            current_res = current_res * 2
            block = nn.ModuleList([ResBlock(filters[i], filters[i+1])])
            for _ in range(depth-1):
                block.append(ResBlock(filters[i+1], filters[i+1]))
            if current_res in attn_resolutions:
                block.append(SpatialAttention(filters[i+1]))
            block.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            self.up.append(block)

        self.norm = nn.GroupNorm(num_groups=32, num_channels=filters[-1], eps=1e-6, affine=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_out = nn.Conv2d(filters[-1], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = rearrange(x, "B z (H W) -> B z H W", H=self.latent_hw, W=self.latent_hw)
        B, D, H, W = x.shape
        assert (H, W) == (self.latent_hw, self.latent_hw), f"expected input shape {(self.latent_hw, self.latent_hw)}, got {(H, W)}"
        assert D == self.z_channels, f"expected {self.z_channels} channels, got {D}"
        x = self.conv_in(x)
        x = self.mid(x)
        for block in self.up:
            for layer in block:
                x = layer(x)
        x = self.norm(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return torch.tanh(x)
    
if __name__ == "__main__":
    hw = 128
    filters = [32, 64, 128, 256]

    diagonal = Encoder(4, hw, filters, [], 2)(torch.randn(4, 3, 128, 128))
    print(diagonal.sample().shape, diagonal.logvar.shape, diagonal.mean.shape, diagonal.mode().shape)
    reconstructed = Decoder(4, hw // (2**(len(filters))),  filters, [], 2)(diagonal.sample())
    print(reconstructed.shape)