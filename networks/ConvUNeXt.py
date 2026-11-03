
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptationBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super(DomainAdaptationBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze: 全局池化 -> [B, C, 1, 1] -> flatten -> [B, C]
        w_vec = self.squeeze(x).flatten(1)                # [B, C]
        # Excitation: [B, C] -> [B, C]
        w_vec = self.excite(w_vec)
        # 通道权重: [B, C] -> [B, C, 1, 1]
        w_vec = self.sigmoid(w_vec).view(b, c, 1, 1)
        # 残差通道注意力
        return x + x * w_vec

class EarlySpatialAttention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EarlySpatialAttention, self).__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, in_channels, H, W]
        # 空间注意力 mask
        mask = self.spatial_gate(x)               # [B, 1, H, W]
        # 增强后的特征
        feat = self.feat_conv(x)                  # [B, out_channels, H, W]
        # mask 作用：抑制不重要的空间区域
        feat = feat * mask
        return feat

class Conv(nn.Module):

    def __init__(self, dim):
        super(Conv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1,
                                groups=dim, padding_mode='reflect')
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.norm2(x)
        x = self.act2(residual + x)
        return x


class Down(nn.Sequential):
    """下采样块：BN -> stride-2 conv -> Conv 块(s)"""

    def __init__(self, in_channels, out_channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )


class GatedSkipConnection(nn.Module):

    def __init__(self, x_decoder_channels, x_skip_channels, out_channels):
        super(GatedSkipConnection, self).__init__()
        # C  : 解码器侧通道数（上采样后用于门控计算）
        # C2 : 跳跃连接通道数
        C = x_decoder_channels
        C2 = x_skip_channels

        # 解码器侧 BN + 上采样（只作用在解码器侧特征上）
        self.norm = nn.BatchNorm2d(C)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 显式可信度门控（基于上采样后的解码器特征）
        self.gate_score = nn.Sequential(
            nn.Conv2d(C, max(C // 4, 8), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(C // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(C // 4, 8), 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # skip 投影到 C 通道（门控注意力需要相同通道数）
        self.proj_skip = nn.Conv2d(C2, C, kernel_size=1, bias=False)
        # 门控注意力后投影回 C2 通道
        self.proj_skip_back = nn.Conv2d(C, C2, kernel_size=1, bias=False)

        # ConvUNeXt 风格门控注意力（gate 基于解码器侧，作用于投影后的 skip）
        self.conv_gate = nn.Linear(C, 3 * C)
        self.conv_linear1 = nn.Linear(C, C)
        self.conv_linear2 = nn.Linear(C, C)

        # 1x1 融合 + Conv 块
        self.conv1x1 = nn.Conv2d(C + C2, out_channels, kernel_size=1)
        self.conv = Conv(out_channels)

    def forward(self, x_decoder: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        # x_decoder: 解码器侧特征（小尺寸）
        # x_skip   : 跳跃连接特征（大尺寸）
        x_dec = self.norm(x_decoder)
        x_dec_up = self.up(x_dec)                             # [B, C, 2H, 2W]

        # ---------- 投影 skip 到与 x_dec_up 相同的通道数 ----------
        x_skip_proj = self.proj_skip(x_skip)                  # [B, C, Hs, Ws]

        # ---------- 把 skip 对齐到 x_dec_up 的空间尺寸 ----------
        if x_skip_proj.shape[2:] != x_dec_up.shape[2:]:
            x_skip_proj = F.interpolate(x_skip_proj, size=x_dec_up.shape[2:],
                                         mode='bilinear', align_corners=True)

        # 尺寸兜底（正常情况下已经一致）
        diff_y = x_skip_proj.size(2) - x_dec_up.size(2)
        diff_x = x_skip_proj.size(3) - x_dec_up.size(3)
        if diff_y > 0 or diff_x > 0:
            x_dec_up = F.pad(x_dec_up, [diff_x // 2, diff_x - diff_x // 2,
                                         diff_y // 2, diff_y - diff_y // 2])
        elif diff_y < 0 or diff_x < 0:
            crop_y1 = (-diff_y) // 2
            crop_y2 = crop_y1 + x_skip_proj.size(2)
            crop_x1 = (-diff_x) // 2
            crop_x2 = crop_x1 + x_skip_proj.size(3)
            x_dec_up = x_dec_up[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        gate_w = self.gate_score(x_dec_up)                    # [B, 1, H, W]
        x_dec_gated = x_dec_up * gate_w                       # [B, C, H, W]

        B, C, H, W = x_dec_gated.shape
        x_dec_perm = x_dec_gated.permute(0, 2, 3, 1)          # [B, H, W, C]
        x_skip_perm = x_skip_proj.permute(0, 2, 3, 1)         # [B, H, W, C]

        gate_out = self.conv_gate(x_dec_perm).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate_out[0], gate_out[1], gate_out[2]
        x_skip_perm = (torch.sigmoid(self.conv_linear1(g1 + x_skip_perm)) * x_skip_perm
                       + torch.sigmoid(g2) * torch.tanh(g3))
        x_skip_perm = self.conv_linear2(x_skip_perm)          # [B, H, W, C]

        x_skip_perm = x_skip_perm.permute(0, 3, 1, 2)         # [B, C, H, W]
        x_skip_gated = self.proj_skip_back(x_skip_perm)       # [B, C2, H, W]

        x_cat = torch.cat([x_dec_gated, x_skip_gated], dim=1)  # [B, C+C2, H, W]
        x = self.conv1x1(x_cat)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    """输出层：1x1 conv"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class ConvUNeXt(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 n_classes: int = 1,
                 bilinear: bool = True,
                 base_channels: int = 32):
        super(ConvUNeXt, self).__init__()
        self.in_channels = in_channels
        self.num_classes = n_classes
        self.bilinear = bilinear

        base_c = base_channels
        factor = 2 if bilinear else 1

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            Conv(base_c)
        )
        self.esa = EarlySpatialAttention(base_c, base_c)

        # ---- 编码器 + 域适应块 ----
        # x1: base_c        @ H
        # x2: base_c*2      @ H/2
        # x3: base_c*4      @ H/4
        # x4: base_c*8      @ H/8
        # x5: base_c*16//f  @ H/16
        self.down1 = Down(base_c, base_c * 2)
        self.adapt1 = DomainAdaptationBlock(base_c * 2)

        self.down2 = Down(base_c * 2, base_c * 4)
        self.adapt2 = DomainAdaptationBlock(base_c * 4)

        self.down3 = Down(base_c * 4, base_c * 8, layer_num=3)
        self.adapt3 = DomainAdaptationBlock(base_c * 8)

        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.adapt4 = DomainAdaptationBlock(base_c * 16 // factor)

        # 参数顺序：x_decoder_channels（解码器侧，小尺寸）, x_skip_channels（skip，大尺寸）, out_channels
        # up1: decoder = x5 (base_c*16//factor @ H/16), skip = x4 (base_c*8 @ H/8)   -> out base_c*8  @ H/8
        # up2: decoder = up1 out (base_c*8 @ H/8),   skip = x3 (base_c*4 @ H/4)       -> out base_c*4  @ H/4
        # up3: decoder = up2 out (base_c*4 @ H/4),   skip = x2 (base_c*2 @ H/2)       -> out base_c*2  @ H/2
        # up4: decoder = up3 out (base_c*2 @ H/2),   skip = x1 (base_c   @ H)         -> out base_c    @ H
        self.up1 = GatedSkipConnection(x_decoder_channels=base_c * 16 // factor,
                                        x_skip_channels=base_c * 8,
                                        out_channels=base_c * 8)
        self.up2 = GatedSkipConnection(x_decoder_channels=base_c * 8,
                                        x_skip_channels=base_c * 4,
                                        out_channels=base_c * 4)
        self.up3 = GatedSkipConnection(x_decoder_channels=base_c * 4,
                                        x_skip_channels=base_c * 2,
                                        out_channels=base_c * 2)
        self.up4 = GatedSkipConnection(x_decoder_channels=base_c * 2,
                                        x_skip_channels=base_c,
                                        out_channels=base_c)

        self.out_conv = OutConv(base_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x1 = self.in_conv(x)
        x1 = self.esa(x1)                                     # base_c        @ H

        # 编码器 + 域适应
        x2 = self.down1(x1)
        x2 = self.adapt1(x2)                                  # base_c*2      @ H/2

        x3 = self.down2(x2)
        x3 = self.adapt2(x3)                                  # base_c*4      @ H/4

        x4 = self.down3(x3)
        x4 = self.adapt3(x4)                                  # base_c*8      @ H/8

        x5 = self.down4(x4)
        x5 = self.adapt4(x5)                                  # base_c*16//f  @ H/16

        # 解码器：统一 self.upN(decoder_out, skip)
        x = self.up1(x5, x4)   # H/16 -> H/8， 与 x4 对齐
        x = self.up2(x,  x3)   # H/8  -> H/4， 与 x3 对齐
        x = self.up3(x,  x2)   # H/4  -> H/2， 与 x2 对齐
        x = self.up4(x,  x1)   # H/2  -> H，   与 x1 对齐

        logits = self.out_conv(x)
        return logits

def convunext_v2(base_channels=32):
    return ConvUNeXt(in_channels=3, n_classes=1, base_channels=base_channels)


if __name__ == '__main__':
    model = ConvUNeXt(in_channels=3, n_classes=2, base_channels=48)

    x = torch.randn(2, 3, 256, 256)

    model.train()
    logits = model(x)
    print(f"[训练] logits 形状: {logits.shape}")   # 期望 [2, 2, 256, 256]

    model.eval()
    with torch.no_grad():
        mask = model(x)
    print(f"[推理] 掩码形状: {mask.shape}")        # 期望 [2, 2, 256, 256]

    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f}M")
