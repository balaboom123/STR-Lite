from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class ViTTinyEncoder(nn.Module):
    """MAE-compatible tiny ViT encoder with fixed sin-cos positional embeddings."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: int | Sequence[int] = (4, 8),
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        patch_size = self._normalize_patch_size(patch_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
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
        self.num_features = embed_dim

        self._init_weights()

    @staticmethod
    def _normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, Sequence) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        raise ValueError(f"patch_size must be int or length-2 sequence, got {patch_size}")

    def _init_weights(self):
        grid_h, grid_w = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_h=grid_h,
            grid_w=grid_w,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode image patches. mask is accepted for API compatibility but ignored
        (ViT-tiny encoder always processes all patches)."""
        bsz = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 1:, :]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy alias for forward_features without mask."""
        mask = torch.zeros(x.shape[0], self.patch_embed.num_patches, device=x.device, dtype=torch.bool)
        return self.forward_features(x, mask)


class ViTSTRAR(nn.Module):
    """Encoder-agnostic autoregressive Transformer decoder for text recognition.

    Follows the HieraOCR architecture: accepts any encoder that exposes
    ``patch_embed.num_patches``, ``num_features``, and
    ``forward_features(images, mask) -> (B, num_patches, dim)``.
    """

    def __init__(
        self,
        encoder,
        vocab_size,
        max_seq_len,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=12,
        decoder_mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if vocab_size is None or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if max_seq_len is None or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")

        self.encoder = encoder
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self._tgt_mask_cache = {}

        encoder_dim = getattr(encoder, "num_features", None)
        if encoder_dim is None:
            raise ValueError("encoder must expose num_features for projection")

        self.encoder_proj = nn.Linear(encoder_dim, decoder_embed_dim)
        self.token_embed = nn.Embedding(self.vocab_size, decoder_embed_dim)
        self.pos_embed = nn.Embedding(self.max_seq_len, decoder_embed_dim)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * decoder_mlp_ratio),
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.output_proj = nn.Linear(decoder_embed_dim, self.vocab_size)

    def _encode(self, images):
        num_patches = self.encoder.patch_embed.num_patches
        mask = torch.zeros(images.shape[0], num_patches, device=images.device, dtype=torch.bool)
        features = self.encoder.forward_features(images, mask)
        return self.encoder_proj(features)

    def _build_tgt_mask(self, seq_len, device):
        key = (device.type, device.index, int(seq_len))
        cached = self._tgt_mask_cache.get(key)
        if cached is not None:
            return cached

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        if len(self._tgt_mask_cache) > 8:
            self._tgt_mask_cache.clear()
        self._tgt_mask_cache[key] = mask
        return mask

    def _run_decoder(self, tgt, memory, tgt_key_padding_mask=None):
        tgt_mask = self._build_tgt_mask(tgt.shape[0], tgt.device)
        return self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_is_causal=True,
        )

    def _decode_one_token_with_cache(self, token_hidden, memory, layer_caches):
        """Incremental decode for one token using per-layer KV cache."""
        x = token_hidden

        for idx, layer in enumerate(self.decoder.layers):
            layer_input = x
            past = layer_caches[idx]
            if past is None:
                kv = layer_input
            else:
                kv = torch.cat([past, layer_input], dim=0)
            layer_caches[idx] = kv

            if layer.norm_first:
                q = layer.norm1(layer_input)
                kv_norm = layer.norm1(kv)
                sa_out = layer.self_attn(
                    q, kv_norm, kv_norm,
                    need_weights=False, is_causal=False,
                )[0]
                x = layer_input + layer.dropout1(sa_out)

                mha_out = layer.multihead_attn(
                    layer.norm2(x), memory, memory,
                    need_weights=False, is_causal=False,
                )[0]
                x = x + layer.dropout2(mha_out)
                x = x + layer._ff_block(layer.norm3(x))
            else:
                sa_out = layer.self_attn(
                    layer_input, kv, kv,
                    need_weights=False, is_causal=False,
                )[0]
                x = layer.norm1(layer_input + layer.dropout1(sa_out))

                mha_out = layer.multihead_attn(
                    x, memory, memory,
                    need_weights=False, is_causal=False,
                )[0]
                x = layer.norm2(x + layer.dropout2(mha_out))
                x = layer.norm3(x + layer._ff_block(x))

        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    def encode(self, images):
        """Encode images to memory. Use with decode() to avoid redundant encoding."""
        return self._encode(images)

    def decode(self, memory, tgt_input, tgt_key_padding_mask=None):
        """Decode from pre-computed encoder memory."""
        if tgt_input is None:
            raise ValueError("tgt_input is required for Transformer decoding")
        if tgt_input.shape[1] > self.max_seq_len:
            raise ValueError(
                f"tgt_input length {tgt_input.shape[1]} exceeds max_seq_len {self.max_seq_len}"
            )

        seq_len = tgt_input.shape[1]
        positions = torch.arange(seq_len, device=tgt_input.device).unsqueeze(0)
        tgt = self.token_embed(tgt_input) + self.pos_embed(positions)
        tgt = self.dropout(tgt)

        tgt = tgt.transpose(0, 1)
        mem = memory.transpose(0, 1)

        decoded = self._run_decoder(tgt, mem, tgt_key_padding_mask=tgt_key_padding_mask)
        decoded = decoded.transpose(0, 1)
        decoded = self.decoder_norm(decoded)
        logits = self.output_proj(decoded)
        return logits

    def forward(self, images, tgt_input, tgt_key_padding_mask=None):
        memory = self.encode(images)
        return self.decode(memory, tgt_input, tgt_key_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, images, bos_id, eos_id, max_len=None):
        """Greedy autoregressive decoding for inference."""
        was_training = self.training
        if was_training:
            self.eval()

        if max_len is None:
            max_len = self.max_seq_len

        B = images.shape[0]
        device = images.device
        memory = self._encode(images).transpose(0, 1)
        layer_caches = [None] * len(self.decoder.layers)

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            pos_idx = generated.shape[1] - 1
            pos = torch.full((1, 1), pos_idx, device=device, dtype=torch.long).expand(B, 1)
            current_token = generated[:, -1:]
            step_hidden = self.token_embed(current_token) + self.pos_embed(pos)
            step_hidden = self.dropout(step_hidden).transpose(0, 1)

            decoded_last = self._decode_one_token_with_cache(
                token_hidden=step_hidden,
                memory=memory,
                layer_caches=layer_caches,
            )
            decoded_last = self.decoder_norm(decoded_last.transpose(0, 1))
            logits = self.output_proj(decoded_last[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            if finished.any():
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        if was_training:
            self.train()
        return generated


def vit_tiny_str_ar_patch4x8(**kwargs):
    encoder_keys = {
        "img_size", "patch_size", "in_chans", "embed_dim", "depth",
        "num_heads", "mlp_ratio", "drop_path_rate",
    }
    decoder_keys = {
        "vocab_size", "max_seq_len", "decoder_embed_dim", "decoder_depth",
        "decoder_num_heads", "decoder_mlp_ratio", "dropout",
    }

    defaults = dict(
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        decoder_embed_dim=192,
        decoder_depth=1,
        decoder_num_heads=12,
        decoder_mlp_ratio=4.0,
        dropout=0.1,
    )
    defaults.update(kwargs)

    enc_kwargs = {k: defaults[k] for k in encoder_keys if k in defaults}
    enc_kwargs["norm_layer"] = partial(nn.LayerNorm, eps=1e-6)
    encoder = ViTTinyEncoder(**enc_kwargs)

    dec_kwargs = {k: defaults[k] for k in decoder_keys if k in defaults}
    return ViTSTRAR(encoder=encoder, **dec_kwargs)
