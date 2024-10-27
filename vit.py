import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class Attention(nn.Module):
    """
    Attention for images
    Input:
        - x: (B, C, H, W), already patched
    Output:
        - x: (B, C, H, W)
    """
    def __init__(
            self,
            in_channels: int,
            num_heads: int,
            dropout: float = 0.0
        ):
        assert in_channels % num_heads == 0, \
            f"in_channels(got {in_channels}) must be divisible by num_heads(got {num_heads})"

        super().__init__()
        self.qkv_transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # self.dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.scale = in_channels ** -0.5
    
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_transform.weight)

    def drop_attn(self, attn):
        B, h, N, _ = attn.shape
        mask = torch.rand((B, h, N, N), device=attn.device) < self.dropout
        attn[mask] = float('-inf')
        return attn


    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_transform(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'B (h d) H W -> B h (H W) d', h=self.num_heads), (q, k, v))
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, h, H*W, H*W)
        attn = self.drop_attn(attn)
        x = self.softmax(attn) @ v
        x = rearrange(x, 'B h (H W) d -> B (h d) H W', H=H, W=W)
        return x


class FFN(nn.Module):
    """
    Feed Forward Network for images
    Input:
        - x: (B, C, H, W)
    Output:
        - x: (B, C, H, W)
    """
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            dropout: float = 0.0
        ):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Dropout(dropout),
        )
    
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.ffn(x)


def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def pair_divisible(x, y):
    x = make_pair(x)
    y = make_pair(y)
    return x[0] % y[0] == 0 and x[1] % y[1] == 0


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    Input:
        - x: (B, c, h, w)
    Output:
        - y: (B, num_classes) or (B, c, h, w)
    """
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            img_size: tuple[int, int],
            patch_size: int | tuple[int, int],
            d_model: int,
            num_heads: int,
            num_layers: int,
            ffn_hidden_channels: int,
            dropout: float = 0.0,
            classifier: bool = True,
        ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = make_pair(img_size)
        self.patch_size = make_pair(patch_size)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_hidden_channels = ffn_hidden_channels
        self.dropout = dropout

        assert d_model % num_heads == 0, \
            f"d_model(got {d_model}) must be divisible by num_heads(got {num_heads})"
        assert pair_divisible(self.img_size, self.patch_size), \
            f"img_size(got {self.img_size}) must be divisible by patch_size(got {self.patch_size})"
        H = self.img_size[0] // self.patch_size[0]
        W = self.img_size[1] // self.patch_size[1]
        
        self.resize = lambda x: F.interpolate(x, size=self.img_size, mode='bilinear')
        self.patchify = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(d_model),
                Attention(d_model, num_heads, dropout),
                nn.BatchNorm2d(d_model),
                FFN(d_model, ffn_hidden_channels, dropout),
            ])
            for _ in range(num_layers)
        ]) # (B, d_model, H, W)

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * H * W, (d_model*H*W) // 2),
            nn.GELU(),
            nn.Linear((d_model*H*W) // 2, num_classes),
        ) if classifier else nn.Sequential(
            nn.ConvTranspose2d(d_model, in_channels, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patchify.weight)
        for m in self.header:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.resize(x)
        x = self.patchify(x)
        for block in self.blocks:
            bn0, attn, bn1, ffn = block
            x = x + attn(bn0(x))
            x = x + ffn(bn1(x))
        x = self.header(x)
        return x


if __name__ == "__main__":
    vit = VisionTransformer(
        num_classes=10,
        in_channels=3,
        img_size=(32, 32),
        patch_size=4,
        d_model=64,
        num_heads=4,
        num_layers=4,
        ffn_hidden_channels=128,
        dropout=0.1,
    )

    x = torch.randn(2, 3, 32, 32)
    y = vit(x)
    print(y.shape) # (2, 10)
    print(y)