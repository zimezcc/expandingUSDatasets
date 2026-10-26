"""
MedNeXt 2D 实现（对齐官方 MIC-DKFZ/MedNeXt 结构）
- 与官方差异修正:
  1. Block 继承结构: MedNeXtDownBlock/MedNeXtUpBlock 继承 MedNeXtBlock
  2. GroupNorm 改为 InstanceNorm 等效: GroupNorm(in_channels, in_channels)
  3. Stem 改为 1x1 卷积 (官方用 kernel_size=1)
  4. Skip connection 改为相加 (官方用 x_res + x_up, 非 concat)
  5. UpBlock 非对称 padding (1,0,1,0) 替代 output_padding
  6. 加入可选 GRN (全局响应归一化)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtBlock(nn.Module):
    """MedNeXt 基础块（2D 版，对齐官方 blocks.py）"""

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=True, norm_type='group', n_groups=None, dim='2d', grn=False):
        super().__init__()

        self.do_res = do_res
        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d

        # First convolution: Depthwise Conv
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer — 官方默认 GroupNorm(num_groups=in_channels, ...)
        # 这等价于 InstanceNorm (每个通道独立归一化)
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = nn.GroupNorm(1, in_channels)  # fallback
        else:
            self.norm = nn.GroupNorm(in_channels, in_channels)

        # Second convolution: Expansion (1x1)
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU
        self.act = nn.GELU()

        # Third convolution: Compression (1x1)
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GRN (Global Response Normalization, ConvNeXt V2)
        self.grn = grn
        if grn:
            if self.dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.grn:
            if self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            else:
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

        x1 = self.conv3(x1)

        if self.do_res:
            x1 = x + x1

        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """MedNeXt 下采样块（2D 版，继承 MedNeXtBlock，覆写 conv1 为 stride=2）"""

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='2d', grn=False):
        # 父类初始化，但 do_res=False (内部残差由父类控制)
        super().__init__(
            in_channels, out_channels, exp_r, kernel_size,
            do_res=False, norm_type=norm_type, dim=dim, grn=grn
        )

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        # 下采样的额外残差 (resample_do_res)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        # 覆写 conv1 为 stride=2 的深度卷积
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)  # 执行父类完整流程 (DWConv stride=2 -> Norm -> Expand -> GELU -> Compress)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """MedNeXt 上采样块（2D 版，继承 MedNeXtBlock，用 ConvTranspose2d）"""

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='2d', grn=False):
        super().__init__(
            in_channels, out_channels, exp_r, kernel_size,
            do_res=False, norm_type=norm_type, dim=dim, grn=grn
        )

        self.resample_do_res = do_res
        self.dim = dim

        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d

        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        # 覆写 conv1 为 ConvTranspose2d stride=2
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)

        # 官方: 非对称 padding (1,0,1,0) 以匹配尺寸
        if self.dim == '2d':
            x1 = F.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = F.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = F.pad(res, (1, 0, 1, 0))
            elif self.dim == '3d':
                res = F.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):
    """输出块 (1x1 Conv)"""

    def __init__(self, in_channels, n_classes, dim='2d'):
        super().__init__()
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.conv = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv(x)


class MedNeXt2D(nn.Module):
    """
    MedNeXt 2D 分割网络（对齐官方 MIC-DKFZ/MedNeXt 结构）

    参数:
        in_channels: 输入通道数
        n_classes: 输出类别数（二分类为 1）
        preset: 'S' / 'B' / 'M' / 'L'
        kernel_size: 深度卷积核大小（默认 5，可选 3/5/7）
        deep_supervision: 是否启用深监督（训练时用）
        do_res: 编码器/解码器 block 内部是否使用残差连接
        do_res_up_down: 上下采样 block 是否使用额外残差连接
        grn: 是否启用 GRN (全局响应归一化)
    """

    PRESETS = {
        'S': {'n_channels': 32, 'exp_r': 2, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'B': {'n_channels': 32, 'exp_r': 3, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'M': {'n_channels': 32, 'exp_r': 4, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'L': {'n_channels': 32, 'exp_r': 4, 'block_counts': [3, 3, 3, 3, 3, 3, 3, 3, 3]},
    }

    def __init__(self, in_channels=3, n_classes=1,
                 preset='B', kernel_size=5,
                 deep_supervision=False,
                 do_res=True, do_res_up_down=False, grn=False):
        super().__init__()

        cfg = self.PRESETS[preset]
        n_channels = cfg['n_channels']
        exp_r = cfg['exp_r']
        block_counts = cfg['block_counts']

        self.do_ds = deep_supervision
        self.n_classes = n_classes

        # exp_r 可以是 int 或 list
        if type(exp_r) == int:
            exp_r = [exp_r for _ in range(len(block_counts))]

        # Stem — 官方使用 1x1 卷积
        self.stem = nn.Conv2d(in_channels, n_channels, kernel_size=1)

        # ===== 编码器 =====
        # enc_block_0: n_channels
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels, out_channels=n_channels,
                exp_r=exp_r[0], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[0])
        ])
        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels, out_channels=2 * n_channels,
            exp_r=exp_r[1], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )

        # enc_block_1: 2*n_channels
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=2 * n_channels, out_channels=2 * n_channels,
                exp_r=exp_r[1], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[1])
        ])
        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels, out_channels=4 * n_channels,
            exp_r=exp_r[2], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )

        # enc_block_2: 4*n_channels
        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=4 * n_channels, out_channels=4 * n_channels,
                exp_r=exp_r[2], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[2])
        ])
        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels, out_channels=8 * n_channels,
            exp_r=exp_r[3], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )

        # enc_block_3: 8*n_channels
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=8 * n_channels, out_channels=8 * n_channels,
                exp_r=exp_r[3], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[3])
        ])
        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels, out_channels=16 * n_channels,
            exp_r=exp_r[4], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )

        # ===== Bottleneck: 16*n_channels =====
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=16 * n_channels, out_channels=16 * n_channels,
                exp_r=exp_r[4], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[4])
        ])

        # ===== 解码器 =====
        # up_3 / dec_block_3: 16*n -> 8*n
        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels, out_channels=8 * n_channels,
            exp_r=exp_r[5], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=8 * n_channels, out_channels=8 * n_channels,
                exp_r=exp_r[5], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[5])
        ])

        # up_2 / dec_block_2: 8*n -> 4*n
        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels, out_channels=4 * n_channels,
            exp_r=exp_r[6], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=4 * n_channels, out_channels=4 * n_channels,
                exp_r=exp_r[6], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[6])
        ])

        # up_1 / dec_block_1: 4*n -> 2*n
        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels, out_channels=2 * n_channels,
            exp_r=exp_r[7], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=2 * n_channels, out_channels=2 * n_channels,
                exp_r=exp_r[7], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[7])
        ])

        # up_0 / dec_block_0: 2*n -> n
        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels, out_channels=n_channels,
            exp_r=exp_r[8], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type='group', dim='2d', grn=grn
        )
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels, out_channels=n_channels,
                exp_r=exp_r[8], kernel_size=kernel_size,
                do_res=do_res, norm_type='group', dim='2d', grn=grn
            )
            for _ in range(block_counts[8])
        ])

        # ===== 输出头 =====
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim='2d')

        # 深监督输出头 (官方: 4 个额外头, 对应不同尺度)
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=2 * n_channels, n_classes=n_classes, dim='2d')
            self.out_2 = OutBlock(in_channels=4 * n_channels, n_classes=n_classes, dim='2d')
            self.out_3 = OutBlock(in_channels=8 * n_channels, n_classes=n_classes, dim='2d')
            self.out_4 = OutBlock(in_channels=16 * n_channels, n_classes=n_classes, dim='2d')

        self.block_counts = block_counts

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # ===== 编码器 =====
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        # ===== Bottleneck =====
        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        # ===== 解码器 (skip = 相加, 非拼接) =====
        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        # 最终输出
        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


# ==================== 使用示例 ====================
if __name__ == '__main__':
    model = MedNeXt2D(
        in_channels=1,
        n_classes=1,
        preset='B',
        kernel_size=5,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=False,
        grn=False
    )

    x = torch.randn(2, 1, 256, 256)

    model.train()
    logits = model(x)
    print(f"[训练] logits 形状: {logits.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"[推理] 输出形状: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f}M")
