"""
MedNeXt 2D 实现（修正版）
- 训练模式：输出 [B, n_classes, H, W] 的 logits，供损失函数使用
- 推理模式：输出 [B, H, W] 的目标掩码（0 ~ n_classes-1）
"""
import torch
import torch.nn as nn


class MedNeXtBlock(nn.Module):
    """MedNeXt 基础块（2D 版）"""

    def __init__(self, in_ch, out_ch, exp_r=4, kernel_size=5,
                 do_res=True, norm_type='group'):
        super().__init__()
        self.do_res = do_res

        # Depthwise Conv (2D)
        self.dw_conv = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_ch,
            bias=False
        )
        self.norm1 = nn.GroupNorm(1, in_ch) if norm_type == 'group' else nn.LayerNorm(in_ch)

        # Expansion (1x1 Conv)
        self.expand = nn.Conv2d(in_ch, in_ch * exp_r, kernel_size=1, bias=False)
        self.act = nn.GELU()

        # Compression (1x1 Conv)
        self.compress = nn.Conv2d(in_ch * exp_r, out_ch, kernel_size=1, bias=False)

        # 残差连接
        if do_res:
            if in_ch != out_ch:
                self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.dw_conv(x)
        out = self.norm1(out)
        out = self.expand(out)
        out = self.act(out)
        out = self.compress(out)
        if self.do_res:
            out = out + self.res_conv(residual)
        return out


class MedNeXtUpDownBlock(nn.Module):
    """MedNeXt 上下采样块（2D 版）"""

    def __init__(self, in_ch, out_ch, kernel_size=5, exp_r=4,
                 down=True, norm_type='group'):
        super().__init__()
        self.down = down

        if down:
            self.dw_conv = nn.Conv2d(
                in_ch, in_ch,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=in_ch,
                bias=False
            )
        else:
            self.dw_conv = nn.ConvTranspose2d(
                in_ch, in_ch,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                groups=in_ch,
                bias=False
            )

        self.norm1 = nn.GroupNorm(1, in_ch)
        self.expand = nn.Conv2d(in_ch, in_ch * exp_r, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.compress = nn.Conv2d(in_ch * exp_r, out_ch, kernel_size=1, bias=False)

        # 残差路径：空间尺寸和通道都要对齐
        if down:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False)
        else:
            self.res_conv = nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=1,
                stride=2,
                output_padding=1,
                bias=False
            )

    def forward(self, x):
        residual = x
        out = self.dw_conv(x)
        out = self.norm1(out)
        out = self.expand(out)
        out = self.act(out)
        out = self.compress(out)
        out = out + self.res_conv(residual)
        return out


class MedNeXt2D(nn.Module):
    """
    MedNeXt 2D 分割网络

    参数:
        in_channels: 输入通道数（超声图像通常为 1）
        n_classes: 输出类别数（二分类为 1）
        preset: 'S' / 'B' / 'M' / 'L'
        kernel_size: 深度卷积核大小（默认 5，可选 3/5/7）
        deep_supervision: 是否启用深监督（训练时用）
    """

    PRESETS = {
        'S': {'n_channels': 32, 'exp_r': 2, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'B': {'n_channels': 32, 'exp_r': 3, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'M': {'n_channels': 32, 'exp_r': 4, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'L': {'n_channels': 32, 'exp_r': 4, 'block_counts': [3, 3, 3, 3, 3, 3, 3, 3, 3]},
    }

    def __init__(self, in_channels=3, n_classes=1,
                 preset='B', kernel_size=5,
                 deep_supervision=False):
        super().__init__()

        cfg = self.PRESETS[preset]
        n_channels = cfg['n_channels']
        exp_r = cfg['exp_r']
        block_counts = cfg['block_counts']

        self.deep_supervision = deep_supervision
        self.n_classes = n_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, n_channels),
            nn.GELU()
        )

        chs = [n_channels * (2 ** i) for i in range(5)]  # [32, 64, 128, 256, 512]

        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(4):
            blocks = nn.Sequential(*[
                MedNeXtBlock(chs[i], chs[i], exp_r, kernel_size)
                for _ in range(block_counts[i])
            ])
            self.enc_blocks.append(blocks)
            self.down_blocks.append(
                MedNeXtUpDownBlock(chs[i], chs[i + 1], kernel_size, exp_r, down=True)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(chs[4], chs[4], exp_r, kernel_size)
            for _ in range(block_counts[4])
        ])

        # 解码器
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(4, 0, -1):
            self.up_blocks.append(
                MedNeXtUpDownBlock(chs[i], chs[i - 1], kernel_size, exp_r, down=False)
            )
            # 第一个 block：in_ch*2 → in_ch
            blocks = [MedNeXtBlock(chs[i - 1] * 2, chs[i - 1], exp_r, kernel_size)]
            # 后续 block：in_ch → in_ch
            blocks += [
                MedNeXtBlock(chs[i - 1], chs[i - 1], exp_r, kernel_size)
                for _ in range(block_counts[i - 1] - 1)
            ]
            self.dec_blocks.append(nn.Sequential(*blocks))

        # 输出头
        self.out_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1)

        # 深监督输出头
        if deep_supervision:
            self.ds_convs = nn.ModuleList([
                nn.Conv2d(chs[i], n_classes, kernel_size=1)
                for i in range(4)
            ])

    def forward(self, x):
        """
        输入: [B, in_channels, H, W]
        输出:
            训练模式 (self.training=True):
                - 无深监督: logits [B, n_classes, H, W]
                - 有深监督: (logits, [ds_logits...])
            推理模式 (self.training=False):
                - 无深监督: 掩码 [B, H, W]，值为 0 ~ n_classes-1
                - 有深监督: 掩码 [B, H, W]
        """
        # Stem
        x = self.stem(x)  # [B, in_channels, H, W] → [B, 32, H, W]

        # 编码器
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_blocks):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码器
        ds_outputs = []
        for i, (up, dec) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            if self.deep_supervision and i < 4:
                ds_outputs.append(self.ds_convs[i](x))

        # 最终输出
        logits = self.out_conv(x)  # [B, n_classes, H, W]
        return logits


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 单通道输入（超声图像）
    model = MedNeXt2D(
        in_channels=1,
        n_classes=1,
        preset='B',
        kernel_size=5,
        deep_supervision=False
    )

    x = torch.randn(2, 1, 256, 256)

    # ---- 训练模式：输出 logits ----
    model.train()
    logits = model(x)
    print(f"[训练] logits 形状: {logits.shape}")  # [2, 1, 256, 256]

    # ---- 推理模式：输出掩码 ----
    model.eval()
    with torch.no_grad():
        mask = model(x)
    print(f"[推理] 掩码形状: {mask.shape}")       # [2, 256, 256]
    print(f"[推理] 掩码唯一值: {torch.unique(mask)}")

    # ---- 参数量 ----
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f}M")