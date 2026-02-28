from functools import partial

import torch
import torch.nn as nn

from src.models.vit_str_ar import ViTTinyEncoder


class ViTTinySTRCTC(nn.Module):
    """ViT encoder + Transformer sequence model + CTC head."""

    def __init__(
        self,
        num_classes: int,
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        pool_height: str = "mean",
        seq_depth: int = 2,
        seq_num_heads: int = 12,
        seq_dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ViTTinyEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=drop_path_rate,
        )

        self.pool_height = pool_height

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=seq_num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=seq_dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=seq_depth)
        self.sequence_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        torch.nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_tokens(x)
        bsz, _, dim = tokens.shape

        h_grid, w_grid = self.encoder.patch_embed.grid_size
        tokens_2d = tokens.view(bsz, h_grid, w_grid, dim)

        if self.pool_height == "mean":
            seq = tokens_2d.mean(dim=1)
        elif self.pool_height == "max":
            seq = tokens_2d.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pool_height mode: {self.pool_height}")

        seq = seq.permute(1, 0, 2)
        seq = self.sequence_transformer(seq)
        seq = self.sequence_norm(seq)

        logits = self.head(seq)
        return logits


def vit_tiny_str_ctc_patch4(**kwargs):
    defaults = dict(
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        seq_depth=2,
        seq_num_heads=12,
    )
    defaults.update(kwargs)
    return ViTTinySTRCTC(**defaults)
