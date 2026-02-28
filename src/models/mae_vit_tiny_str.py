from functools import partial
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Tiny MAE for scene text with original fixed sin-cos position embeddings."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: int | Sequence[int] = (4, 8),
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 1,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.img_size = tuple(img_size)
        self.patch_size = self._normalize_patch_size(patch_size)
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        dpr_dec = torch.linspace(0, drop_path_rate, decoder_depth).tolist()
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=dpr_dec[i],
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    @staticmethod
    def _normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, Sequence) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        raise ValueError(f"patch_size must be int or length-2 sequence, got {patch_size}")

    def initialize_weights(self):
        grid_h, grid_w = self.patch_embed.grid_size

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_h=grid_h,
            grid_w=grid_w,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            grid_h=grid_h,
            grid_w=grid_w,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p_h, p_w = self.patch_size
        n, c, h, w = imgs.shape
        if h % p_h != 0 or w % p_w != 0:
            raise ValueError(f"Input shape {(h, w)} not divisible by patch size {(p_h, p_w)}")

        h_grid = h // p_h
        w_grid = w // p_w
        x = imgs.reshape(n, c, h_grid, p_h, w_grid, p_w)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(n, h_grid * w_grid, p_h * p_w * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p_h, p_w = self.patch_size
        h_grid, w_grid = self.patch_embed.grid_size
        n = x.shape[0]

        if x.shape[1] != h_grid * w_grid:
            raise ValueError(f"Token length {x.shape[1]} does not match grid {(h_grid, w_grid)}")

        x = x.reshape(n, h_grid, w_grid, p_h, p_w, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(n, self.in_chans, h_grid * p_h, w_grid * p_w)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_tiny_patch4_str(**kwargs):
    defaults = dict(
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=96,
        decoder_depth=1,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=0.0,
    )
    defaults.update(kwargs)
    return MaskedAutoencoderViT(**defaults)
