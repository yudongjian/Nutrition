import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        x1 = x1.view(b, c, -1).permute(0, 2, 1)
        x2 = x2.view(b, c, -1).permute(0, 2, 1)

        attn_output, _ = self.attn(x1, x2, x2)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return attn_output


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate = self.gate(x1 + x2)
        return gate * x1 + (1 - gate) * x2


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x



class FeatureFusionNetwork222(nn.Module):
    def __init__(self, dropout=0.1):
        super(FeatureFusionNetwork222, self).__init__()

        # 特征提取
        self.feature1_conv = nn.Conv2d(192, 512, kernel_size=3, stride=1, padding=1)
        self.feature1_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature2_conv = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.feature2_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature3_conv = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.feature3_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature4_conv = nn.Conv2d(768 * 2, 512, kernel_size=3, stride=1, padding=1)

        # 融合模块
        self.cross_attn1 = CrossAttentionFusion(512)
        self.gated_fusion1 = GatedFusion(512)
        self.cross_attn2 = CrossAttentionFusion(512)
        self.gated_fusion2 = GatedFusion(512)
        self.rgb_fusion = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.semantic_fusion = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # 注意力 & Transformer
        self.attention = AttentionFusion(1024)
        self.transformer_encoder = TransformerEncoder(dim=1024)

        # 输出层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.batchnorm1d = nn.BatchNorm1d(1024)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x1, x2, x3, x4):
        # 处理特征
        x1 = self.feature1_pool(self.feature1_conv(x1))
        x2 = self.feature2_pool(self.feature2_conv(x2))
        x3 = self.feature3_pool(self.feature3_conv(x3))
        x4 = self.feature4_conv(x4)

        # 融合
        rgb_fused = self.cross_attn1(x1, x2)
        rgb_fused = self.gated_fusion1(rgb_fused, x2)
        rgb_fused = self.rgb_fusion(rgb_fused)

        semantic_fused = self.cross_attn2(x3, x4)
        semantic_fused = self.gated_fusion2(semantic_fused, x4)
        semantic_fused = self.semantic_fusion(semantic_fused)

        # 全局融合
        fused_features = torch.cat([rgb_fused, semantic_fused], dim=1)
        fused_features = self.attention(fused_features)

        # 输出
        fused_features = self.global_pool(fused_features)
        fused_features = fused_features.view(fused_features.size(0), -1)

        fused_features = self.dropout(fused_features)
        output = self.fc(fused_features)

        return output

