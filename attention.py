import torch
import torch.nn as nn
import torch.nn.functional as F
from RoPE import apply_angles_2d, generate_angles_2d
from einops import rearrange


class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention Module for 2D feature maps.
    First, we compute normalised query, key, and value tensors using a Conv2d:
    - A 3x3 kernel if Linear=False
    - A 1x1 kernel if Linear=True
    Then, we apply scaled dot-product attention across the spatial dimensions.
    Finally, we project the output back to the original embedding dimension.
    """
    def __init__(self, emb_dim, Linear=False, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(num_groups=32, eps=1e-6, affine=True, num_channels=emb_dim)
        self.qkv = nn.Conv2d(emb_dim, 3*emb_dim, kernel_size=3 if not Linear else 1, stride=1, padding=1 if not Linear else 0, bias=False)
        self.proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % 32 == 0, "Channels must be divisible by Groups (32)"
        assert C % self.n_heads == 0, "Channels must be divisible by number of heads"
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        # to 1D per head
        q = rearrange(q, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)
        k = rearrange(k, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)
        v = rearrange(v, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)

        dx = F.scaled_dot_product_attention(q, k, v)
        dx = rearrange(dx, "B h (H W) D -> B (h D) H W", H=H, W=W, h=self.n_heads)
        dx = self.proj(dx)
        return x + dx

class Attention(nn.Module):
    """
    Self-Attention Module for 1D token sequences.
    First, we compute normalised query, key, and value tensors using a Linear layer
    Then, we apply scaled dot-product attention across the token dimensions.
    Finally, we project the output back to the original embedding dimension.
    """
    def __init__(self, H,W, emb_dim, n_heads=8):
        super().__init__()
        self.H = H
        self.W = W
        self.n_heads = n_heads
        head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)
        self.apply_angles_2d = apply_angles_2d
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.register_buffer("freq", generate_angles_2d(H, W, head_dim), persistent=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # to 2D
        q = rearrange(q, "B (H W) (h D) -> B h H W D", H=self.H, W=self.W, h=self.n_heads)
        k = rearrange(k, "B (H W) (h D) -> B h H W D", H=self.H, W=self.W, h=self.n_heads)

        q = apply_angles_2d(q, self.freq)
        k = apply_angles_2d(k, self.freq)

        # to 1D
        q = rearrange(q, "B h H W D -> B h (H W) D", H=self.H, W=self.W, h=self.n_heads)
        k = rearrange(k, "B h H W D -> B h (H W) D", H=self.H, W=self.W, h=self.n_heads)
        v = rearrange(v, "B N (h D) -> B h N D", N=self.H*self.W, h=self.n_heads)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B h N D -> B N (h D)")
        x = self.proj(x)
        return x

class ViTBlock(nn.Module):
    """
    A Transformer Block for the Vision Transformer Autoencoder - Self-Attention + MLP
    """
    def __init__(self, H, W, emb_dim, n_heads=8, dropout=0.1):
        self.H, self.W, self.emb_dim = H, W, emb_dim
        super().__init__()
        self.attn = nn.Sequential(nn.LayerNorm(emb_dim),
                                Attention(H,W,emb_dim,n_heads=n_heads))
        self.MLP = nn.Sequential(nn.LayerNorm(emb_dim),
                                nn.Linear(emb_dim, emb_dim*4, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(emb_dim*4, emb_dim, bias=True),
                                nn.Dropout(dropout))
    def forward(self, x):
        assert x.ndim == 3, f"Expected shape [B, N, D], but got shape {x.shape}. You probably passed [B, H, W, D] instead."
        assert x.shape == torch.Size([x.shape[0], self.H * self.W, self.emb_dim]), f"Expected shape [B, N, D] -> {torch.Size([x.shape[0], self.H * self.W, self.emb_dim])}, got {x.shape}"
        x = x + self.attn(x)
        x = x + self.MLP(x)
        return x
  
# Sanity Check :)
if __name__ == "__main__":
    print(SpatialAttention(64, n_heads=8)(torch.randn(32, 64, 16, 16)).shape)
    print(ViTBlock(64,64,384)(torch.randn(1, 64**2, 384)).shape)