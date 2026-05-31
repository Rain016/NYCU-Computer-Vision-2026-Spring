"""
PromptIR: Prompting for All-in-One Blind Image Restoration
Backbone: Restormer (MDTA + GDFN transformer blocks)
Prompts: Learned codebook entries weighted by global feature statistics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # [B, C, H, W]


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer)."""

    def __init__(self, channels, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, 3, 1, 1,
            groups=channels * 3, bias=bias
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        head_c = c // self.num_heads

        qkv = self.qkv_dw(self.qkv(x))          # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to [B, heads, head_c, H*W]
        q = q.reshape(b, self.num_heads, head_c, h * w)
        k = k.reshape(b, self.num_heads, head_c, h * w)
        v = v.reshape(b, self.num_heads, head_c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: [B, heads, head_c, head_c]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                          # [B, heads, head_c, H*W]
        out = out.reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-forward Network (Restormer)."""

    def __init__(self, channels, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(channels * expansion)
        self.proj_in = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, 1, 1,
            groups=hidden * 2, bias=bias
        )
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, num_heads, bias)
        self.norm2 = LayerNorm(channels)
        self.ffn = GDFN(channels, expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Prompt Generation Block
# ---------------------------------------------------------------------------

class PromptGenBlock(nn.Module):
    """
    Generates a spatial prompt from the encoder feature at this scale.
    prompt_param: learned codebook [1, L, D, S, S]
    linear_layer: projects global pooling [B, in_dim] -> weights [B, L]
    """

    def __init__(self, in_dim, prompt_dim=128, prompt_len=5, prompt_size=32):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.randn(1, prompt_len, prompt_dim,
                        prompt_size, prompt_size) * 0.02
        )
        self.linear = nn.Linear(in_dim, prompt_len)
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # [B, in_dim] -> [B, prompt_len]
        weights = F.softmax(self.linear(x.mean(dim=(-2, -1))), dim=1)
        # [B, prompt_len, 1, 1, 1] * [B, prompt_len, D, S, S] -> [B, D, S, S]
        prompt = (
            weights[:, :, None, None, None]
            * self.prompt_param.expand(b, -1, -1, -1, -1)
        ).sum(dim=1)
        prompt = F.interpolate(prompt, size=(
            h, w), mode='bilinear', align_corners=False)
        return self.conv(prompt)


# ---------------------------------------------------------------------------
# PromptIR
# ---------------------------------------------------------------------------

class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        expansion=2.66,
        bias=False,
        prompt_dim=128,
        prompt_len=5,
        prompt_size=32,
    ):
        super().__init__()

        # ---- Patch embedding ----
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)

        # ---- Encoder ----
        self.enc1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], expansion, bias)
            for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim * 2, 2, 2, bias=bias)

        self.enc2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], expansion, bias)
            for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 2, 2, bias=bias)

        self.enc3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], expansion, bias)
            for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim * 4, dim * 8, 2, 2, bias=bias)

        # ---- Bottleneck ----
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], expansion, bias)
            for _ in range(num_blocks[3])])

        # ---- Prompt blocks (one per encoder level) ----
        self.prompt3 = PromptGenBlock(
            dim * 4, prompt_dim, prompt_len, prompt_size)
        self.prompt2 = PromptGenBlock(
            dim * 2, prompt_dim, prompt_len, prompt_size)
        self.prompt1 = PromptGenBlock(
            dim,     prompt_dim, prompt_len, prompt_size)

        # Fuse encoder feature + prompt, reduce back to enc channels
        self.fuse3 = nn.Sequential(
            TransformerBlock(dim * 4 + prompt_dim, heads[2], expansion, bias),
            nn.Conv2d(dim * 4 + prompt_dim, dim * 4, 1, bias=bias),
        )
        self.fuse2 = nn.Sequential(
            TransformerBlock(dim * 2 + prompt_dim, heads[1], expansion, bias),
            nn.Conv2d(dim * 2 + prompt_dim, dim * 2, 1, bias=bias),
        )
        self.fuse1 = nn.Sequential(
            TransformerBlock(dim + prompt_dim, heads[0], expansion, bias),
            nn.Conv2d(dim + prompt_dim, dim,     1, bias=bias),
        )

        # ---- Decoder ----
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim * 8, dim * 4, 1, bias=bias))
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dec3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], expansion, bias)
            for _ in range(num_blocks[2])])

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim * 4, dim * 2, 1, bias=bias))
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dec2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], expansion, bias)
            for _ in range(num_blocks[1])])

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim * 2, dim, 1, bias=bias))
        self.dec1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], expansion, bias)
            for _ in range(num_blocks[0])])

        # ---- Refinement & output ----
        self.refine = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], expansion, bias)
            for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        inp = x

        # Encoder
        e1 = self.enc1(self.patch_embed(x))         # [B, d,   H,   W  ]
        e2 = self.enc2(self.down1(e1))               # [B, 2d,  H/2, W/2]
        e3 = self.enc3(self.down2(e2))               # [B, 4d,  H/4, W/4]
        lat = self.latent(self.down3(e3))            # [B, 8d,  H/8, W/8]

        # Decoder with prompt-enhanced skip connections
        p3 = self.fuse3(torch.cat([e3, self.prompt3(e3)], dim=1))
        d3 = self.dec3(self.reduce3(torch.cat([self.up3(lat), p3], dim=1)))

        p2 = self.fuse2(torch.cat([e2, self.prompt2(e2)], dim=1))
        d2 = self.dec2(self.reduce2(torch.cat([self.up2(d3), p2], dim=1)))

        p1 = self.fuse1(torch.cat([e1, self.prompt1(e1)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), p1], dim=1))

        out = self.output(self.refine(d1)) + inp
        return out


def build_model(**kwargs):
    defaults = dict(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        expansion=2.66,
        bias=False,
        prompt_dim=128,
        prompt_len=5,
        prompt_size=32,
    )
    defaults.update(kwargs)
    return PromptIR(**defaults)


if __name__ == '__main__':
    model = build_model()
    n = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n / 1e6:.1f}M')
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f'Input: {x.shape}  Output: {y.shape}')
