"""
ConvUNeXt 2D 实现
基于论文 "ConvUNeXt: An efficient convolution neural network for medical image segmentation" 整理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock2D(nn.Module):
    """ConvNeXt 风格的基础卷积块（2D）"""

    def __init__(self, dim, intermediate_dim=None, kernel_size=7,
                 layer_scale_init_value=1e-6):
        super().__init__()
        intermediate_dim = intermediate_dim or dim * 4

        # Depthwise Conv (大核)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=False
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise (MLP)
        self.pwconv1 = nn.Conv2d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(intermediate_dim, dim, kernel_size=1)

        # Layer Scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)

        # LayerNorm 需要 NHWC 格式
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)

        x = x + residual
        return x


class LightweightAttention(nn.Module):
    """轻量级注意力机制（用于跳跃连接）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DownBlock(nn.Module):
    """下采样块：使用步长卷积代替池化"""

    def __init__(self, in_ch, out_ch, kernel_size=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class ConvUNeXt(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_channels=32):
        super().__init__()
        self.n_classes = n_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 编码器
        self.enc1 = ConvNeXtBlock2D(base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)

        self.enc2 = ConvNeXtBlock2D(base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)

        self.enc3 = ConvNeXtBlock2D(base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvNeXtBlock2D(base_channels * 8),
            ConvNeXtBlock2D(base_channels * 8)
        )

        # 解码器
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)
        self.att3 = LightweightAttention(base_channels * 4)
        self.dec3 = ConvNeXtBlock2D(base_channels * 4)

        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.att2 = LightweightAttention(base_channels * 2)
        self.dec2 = ConvNeXtBlock2D(base_channels * 2)

        self.up1 = UpBlock(base_channels * 2, base_channels)
        self.att1 = LightweightAttention(base_channels)
        self.dec1 = ConvNeXtBlock2D(base_channels)

        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)                    # <<< 新增

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(d3 + self.att3(e3))

        d2 = self.up2(d3)
        d2 = self.dec2(d2 + self.att2(e2))

        d1 = self.up1(d2)
        d1 = self.dec1(d1 + self.att1(e1))

        logits = self.out_conv(d1)          # [B, n_classes, H, W]
        return logits


# ==================== 测试 ====================
if __name__ == '__main__':
    model = ConvUNeXt(in_channels=1, n_classes=2, base_channels=32)

    x = torch.randn(2, 1, 256, 256)

    # 训练模式
    model.train()
    logits = model(x)
    print(f"[训练] logits 形状: {logits.shape}")  # [2, 2, 256, 256]

    # 推理模式
    model.eval()
    with torch.no_grad():
        mask = model(x)
    print(f"[推理] 掩码形状: {mask.shape}")  # [2, 256, 256]

    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f}M")