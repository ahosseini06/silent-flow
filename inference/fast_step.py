"""Incremental (KV-cached) single-token steps for the ESPnet decoder and LM.

ESPnet's `forward_one_step` caches layer *outputs*, so every step re-projects
keys/values for the whole prefix, and the decoder re-projects the full encoder
output for every beam in every layer. Here self-attention K/V are cached per
layer and the cross-attention K/V of the encoder output are computed once per
utterance and shared across the beam. Mathematically identical, much less work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

KV = tuple[torch.Tensor, torch.Tensor]


def _heads(linear, x: torch.Tensor, h: int, dk: int) -> torch.Tensor:
    b, l, _ = x.shape
    return linear(x).view(b, l, h, dk).transpose(1, 2)  # (B, H, L, dk)


def _attend(attn, q_in: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b = q_in.size(0)
    q = _heads(attn.linear_q, q_in, attn.h, attn.d_k)
    out = F.scaled_dot_product_attention(q, k, v)  # (B, H, 1, dk)
    return attn.linear_out(out.transpose(1, 2).reshape(b, 1, attn.h * attn.d_k))


def _self_attn(attn, h: torch.Tensor, past: KV | None) -> tuple[torch.Tensor, KV]:
    k = _heads(attn.linear_k, h, attn.h, attn.d_k)
    v = _heads(attn.linear_v, h, attn.h, attn.d_k)
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    return _attend(attn, h, k, v), (k, v)


def select(cache: list[KV] | None, idx: torch.Tensor) -> list[KV] | None:
    if cache is None:
        return None
    return [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache]


class FastDecoder:
    def __init__(self, decoder):
        self.d = decoder
        self.emb, self.pe = decoder.embed[0], decoder.embed[1]
        assert type(self.pe).__name__ == "PositionalEncoding", type(self.pe)
        assert decoder.normalize_before
        for layer in decoder.decoders:
            assert layer.normalize_before and not layer.concat_after
        self.cross: list[KV] = []

    def start(self, memory: torch.Tensor) -> None:
        """memory: (T, D) encoder output for this utterance."""
        m = memory.unsqueeze(0)
        self.cross = [
            (_heads(l.src_attn.linear_k, m, l.src_attn.h, l.src_attn.d_k),
             _heads(l.src_attn.linear_v, m, l.src_attn.h, l.src_attn.d_k))
            for l in self.d.decoders
        ]

    def step(self, tokens: torch.Tensor, pos: int, cache: list[KV] | None):
        """tokens: (W,) last token of each hypothesis at position `pos`."""
        w = tokens.size(0)
        if self.pe.pe.size(1) <= pos or self.pe.pe.device != tokens.device:
            self.pe.extend_pe(torch.zeros(1, pos + 1, device=tokens.device))
        x = self.emb(tokens).unsqueeze(1) * self.pe.xscale + self.pe.pe[:, pos : pos + 1]
        new_cache = []
        for i, l in enumerate(self.d.decoders):
            a, kv = _self_attn(l.self_attn, l.norm1(x), cache[i] if cache else None)
            new_cache.append(kv)
            x = x + a
            ck, cv = self.cross[i]
            x = x + _attend(l.src_attn, l.norm2(x), ck.expand(w, -1, -1, -1), cv.expand(w, -1, -1, -1))
            x = x + l.feed_forward(l.norm3(x))
        y = self.d.after_norm(x[:, -1])
        return torch.log_softmax(self.d.output_layer(y), dim=-1), new_cache


class FastLM:
    def __init__(self, lm):
        self.lm = lm
        enc = lm.encoder
        assert enc.normalize_before
        for l in enc.encoders:
            assert l.normalize_before and not l.concat_after and not l.macaron_style
            assert l.conv_module is None

    def step(self, tokens: torch.Tensor, cache: list[KV] | None):
        enc = self.lm.encoder
        x = enc.embed(self.lm.embed(tokens).unsqueeze(1))  # no positional encoding in this LM
        new_cache = []
        for i, l in enumerate(enc.encoders):
            a, kv = _self_attn(l.self_attn, l.norm_mha(x), cache[i] if cache else None)
            new_cache.append(kv)
            x = x + a
            x = x + l.ff_scale * l.feed_forward(l.norm_ff(x))
        y = enc.after_norm(x[:, -1])
        return self.lm.decoder(y).log_softmax(dim=-1), new_cache
