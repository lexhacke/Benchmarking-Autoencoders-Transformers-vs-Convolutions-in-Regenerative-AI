import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attention import ViTBlock
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class Encoder(nn.Module):
    """
    Vision Transformer Encoder Module for 2D images.
    - Takes an input image of form B, 3, H, W and encodes it into a latent representation of shape B, D, N.
    """
    def __init__(self, z_channels, hw, emb_dim, patch_size, n_heads=8, dropout=0.1, layers=6, device='cpu'):
        assert emb_dim % n_heads == 0, "Embedding dimension can't be partitioned into n_heads"
        super().__init__()
        self.patchifier = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.Blocks = nn.ModuleList([ViTBlock(hw // patch_size, hw // patch_size, emb_dim, n_heads=n_heads, dropout=dropout, device=device) for _ in range(layers)])
        self.post_norm = nn.LayerNorm(emb_dim)
        self.compress_latent = nn.Linear(emb_dim, z_channels*2)

    def forward(self,x):
        x = self.patchifier(x)
        x = rearrange(x, "B D H W -> B (H W) D") # Flatten to B, N, D
        for vitBlock in self.Blocks:
            x = vitBlock(x)
        x = self.post_norm(x)
        x = self.compress_latent(x)
        # shape is [B HW/p**2 (2 D)] but DiagonalGaussianDistribution expects [B, 2D, HW/p**2]
        return DiagonalGaussianDistribution(rearrange(x, "B N D -> B D N"))

class Decoder(nn.Module):
    """
    Vision Transformer Decoder Module for 1D latent token sequences.
    - Takes an input image of form B, D, N and decodes it into B, 3, H, W image(s).
    """
    def __init__(self, z_channels, hw, emb_dim, patch_size, n_heads=8, dropout=0.1, layers=6, device='cpu'):
        assert emb_dim % n_heads == 0, "Embedding dimension can't be partitioned into n_heads"
        super().__init__()
        self.hw = hw // patch_size
        self.patch_size = patch_size
        self.decompress_latent = nn.Linear(z_channels, emb_dim)
        self.post_norm = nn.LayerNorm(emb_dim)
        self.emb_to_patch = nn.Linear(emb_dim, 3*(patch_size**2))
        self.Blocks = nn.ModuleList([ViTBlock(hw // patch_size, hw // patch_size, emb_dim, n_heads=n_heads, dropout=dropout, device=device) for _ in range(layers)])

    def forward(self, x):
        # x is the latent DiagonalGaussianDistribution sample, shape [B, z, HW/p**2] we need [B, HW/p**2, z]
        x = rearrange(x, "B D N -> B N D")
        x = self.decompress_latent(x)
        for vitBlock in self.Blocks:
            x = vitBlock(x)
        x = self.post_norm(x)
        #shape is [B HW/p**2 (3 p p)]
        x = self.emb_to_patch(x)
        assert x.shape == torch.Size([x.shape[0], self.hw**2, 3*(self.patch_size**2)]), f"Expected shape {torch.Size([x.shape[0], self.hw**2, 3*(self.patch_size**2)])} got {x.shape}"
        x = rearrange(x, "B (H W) (D p1 p2) -> B D (H p1) (W p2)", H=self.hw, W=self.hw, p1=self.patch_size, p2=self.patch_size) # Expand to B, H, W, D
        return torch.tanh(x)
  
if __name__ == "__main__":
    emb_dim = 384
    patch_size = 16
    hw = 128

    diagonal = Encoder(4, hw, emb_dim, patch_size)(torch.randn(4, 3, 128, 128))
    print(diagonal.sample().shape, diagonal.logvar.shape, diagonal.mean.shape, diagonal.mode().shape)
    reconstructed = Decoder(4, hw, emb_dim, patch_size)(diagonal.sample())
    print(reconstructed.shape)