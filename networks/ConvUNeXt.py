"""
ConvUNeXt_v2 — 针对合成数据域偏移问题的改进版本

改进点：
  1. DomainAdaptationBlock（编码器每阶段输出后）：
     - SE (Squeeze-and-Excitation) 通道注意力，对合成数据带来的异常纹理做通道级抑制
     - 残差结构，不影响原始信息通路
  2. GatedSkipConnection（替换原有简单 cat + conv）：
     - 显式建模"跳跃连接特征是否可信"
     - 在合成数据上自动降低不可靠跳跃连接的权重
  3. EarlySpatialAttention（Stem 后）：
     - 对输入图像的空间异常区域做初步增强
     - 帮助后续编码器聚焦于肿瘤区域而非背景纹理
  4. base_channels 默认从 32 提升到 48，提升对域偏移的抵抗能力

与原版完全兼容的接口：
  - 参数名: num_classes -> n_classes, base_c -> base_channels
  - 返回值: 直接返回 logits tensor（而非 Dict）

尺寸约定（重要）：
  - GatedSkipConnection.forward(x_decoder, x_skip)
      x_decoder: 解码器侧特征，空间尺寸较小（将被 self.up 上采样 2 倍）
      x_skip   : 编码器跳跃连接特征，空间尺寸较大（将与上采样后的解码器特征对齐）
  - forward 中四个 up 块统一调用 self.upN(decoder_out, skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 改进1: 域适应块 ====================
class DomainAdaptationBlock(nn.Module):
    """通道级域适应块：SE 注意力 + 残差

    作用：对编码器输出做通道重标定，抑制合成数据引入的异常纹理通道，
    增强对真实图像友好的通道响应。相当于让模型"自动学会"在合成数据上
    忽略那些只在特定 normal 图像背景上才出现的高响应通道。
    """

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


# ==================== 改进2: 早期空间注意力 ====================
class EarlySpatialAttention(nn.Module):
    """Stem 后的早期空间注意力

    作用：对输入图像的空间纹理异常区域做增强/抑制。
    合成数据的问题之一是背景 normal 图像带来的非真实纹理，
    此模块帮助模型在浅层就学会聚焦于肿瘤区域。
    """

    def __init__(self, in_channels, out_channels):
        super(EarlySpatialAttention, self).__init__()
        # 空间注意力：两层 3x3 conv 预测每个空间位置的重要性
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 特征增强
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


# ==================== 原有组件（原版实现） ====================
class Conv(nn.Module):
    """ConvNeXt 风格基础卷积块（官方原版）

    结构: 7x7 depthwise conv (reflect padding) -> BN -> Linear(4C) -> GELU -> Linear(C) -> BN -> 残差+GELU
    """

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


# ==================== 改进3: 带显式门控的跳跃连接 ====================
class GatedSkipConnection(nn.Module):
    """改进的跳跃连接：显式建模"特征可信度"

    语义约定（与 forward 调用一致）：
      forward(x_decoder, x_skip)
        x_decoder: 解码器侧特征，空间尺寸较小，将被 self.up 上采样 2 倍
        x_skip   : 编码器跳跃连接特征，空间尺寸较大，用于与上采样后的解码器特征对齐

    原版 ConvUNeXt 的门控注意力是内嵌在 Up 模块中的，这里将其显式化，
    并加入可学习的"可信度"机制，让模型在合成数据上自动降低不可靠跳跃连接的权重。
    """

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

        # ---------- 显式可信度门控（作用于解码器侧上采样特征） ----------
        gate_w = self.gate_score(x_dec_up)                    # [B, 1, H, W]
        x_dec_gated = x_dec_up * gate_w                       # [B, C, H, W]

        # ---------- ConvUNeXt 风格门控注意力（gate 基于解码器侧，作用于 skip） ----------
        B, C, H, W = x_dec_gated.shape
        x_dec_perm = x_dec_gated.permute(0, 2, 3, 1)          # [B, H, W, C]
        x_skip_perm = x_skip_proj.permute(0, 2, 3, 1)         # [B, H, W, C]

        gate_out = self.conv_gate(x_dec_perm).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate_out[0], gate_out[1], gate_out[2]
        x_skip_perm = (torch.sigmoid(self.conv_linear1(g1 + x_skip_perm)) * x_skip_perm
                       + torch.sigmoid(g2) * torch.tanh(g3))
        x_skip_perm = self.conv_linear2(x_skip_perm)          # [B, H, W, C]

        # 投影回 skip 原始通道数 C2
        x_skip_perm = x_skip_perm.permute(0, 3, 1, 2)         # [B, C, H, W]
        x_skip_gated = self.proj_skip_back(x_skip_perm)       # [B, C2, H, W]

        # ---------- 拼接 + 融合 ----------
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


# ==================== 主模型 ====================
class ConvUNeXt(nn.Module):
    """ConvUNeXt_v2 — 针对合成数据域偏移问题的改进版本

    改进点：
      1. Stem 后增加 EarlySpatialAttention
      2. 每个编码器下采样后增加 DomainAdaptationBlock
      3. 解码器跳跃连接改为 GatedSkipConnection
      4. base_channels 默认 32（与原版保持几何对称性）

    尺寸约定：
      - forward 中四个 up 块统一调用 self.upN(decoder_out, skip)
      - upN 的第一个参数是解码器侧（小尺寸），第二个是 skip（大尺寸）
      - upN 内部对解码器侧做 2 倍上采样，与 skip 对齐

    参数:
      in_channels: 输入通道数（默认 3）
      n_classes: 输出类别数（默认 1）
      bilinear: 是否使用双线性插值上采样（默认 True）
      base_channels: 基础通道数（默认 32）
    """

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

        # ---- Stem + 早期空间注意力 ----
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

        # ---- 解码器 + 门控跳跃连接 ----
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


# ==================== 工厂函数 ====================
def convunext_v2(base_channels=32):
    """默认配置：base_channels=32（与原版几何对称）"""
    return ConvUNeXt(in_channels=3, n_classes=1, base_channels=base_channels)


# ==================== 测试 ====================
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