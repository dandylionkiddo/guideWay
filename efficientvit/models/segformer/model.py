import torch
import torch.nn as nn

from .mix_transformer import MixVisionTransformer
from .segformer_head import SegformerHead

class SegFormer(nn.Module):
    def __init__(self, embed_dims, num_heads, mlp_ratios, qkv_bias, depths, sr_ratios, 
                 num_classes, in_channels=3, embedding_dim=256, dropout_ratio=0.1):
        super(SegFormer, self).__init__()

        self.backbone = MixVisionTransformer(
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            depths=depths,
            sr_ratios=sr_ratios,
            norm_layer=nn.LayerNorm
        )

        self.decode_head = SegformerHead(
            in_channels=self.backbone.embed_dims, # The output channels of backbone stages
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_ratio=dropout_ratio
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decode_head(features)
        
        # During training, the loss function in this project likely expects logits
        # of the same size as the input. The original Segformer upsamples to 1/4 size.
        # We will upsample to the full size here.
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return logits

def segformer_b0(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=256, **kwargs)  # 256

def segformer_b1(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=256, **kwargs)  # 256

def segformer_b2(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=768, **kwargs)  # 768로 변경

def segformer_b3(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=768, **kwargs)  # 768로 변경

def segformer_b4(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=768, **kwargs)  # 768로 변경

def segformer_b5(num_classes=19, **kwargs):
    return SegFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        # num_classes=num_classes, **kwargs)
        num_classes=num_classes, embedding_dim=768, **kwargs)  # 768로 변경
