import torch
import torch.nn as nn
from torchvision.models import resnet50

# Simple fake RFE module (just 3x3 conv for now)
class RFEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        memory = self.encoder(src)  # [HW, B, D]
        hs = self.decoder(tgt, memory)  # [Q, B, D]
        return hs  # [Q, B, D]

class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries=100):
        super().__init__()
        # Backbone
        backbone = resnet50(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.rfe = RFEModule(2048)
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)

        # Transformer + queries
        self.query_embed = nn.Embedding(num_queries, 256)
        self.transformer = TransformerBlock()

        # Prediction heads
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 for "no object"
        self.bbox_embed = nn.Linear(256, 4)

    def forward(self, images):
        # Backbone
        features = self.backbone(images)
        features = self.rfe(features)
        features = self.input_proj(features)

        B, C, H, W = features.shape
        src = features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]

        # Transformer
        hs = self.transformer(src, query_embed)  # [Q, B, C]
        hs = hs.permute(1, 0, 2)  # [B, Q, C]

        return {
            "pred_logits": self.class_embed(hs),   # [B, Q, num_classes+1]
            "pred_boxes": self.bbox_embed(hs).sigmoid()  # [B, Q, 4]
        }
