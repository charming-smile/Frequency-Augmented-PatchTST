"""
Frequency-guided Adaptive PatchTST (first version).

Only the patch generation stage is changed:
    input -> RevIN -> FFT dominant period -> rule-based patch_len -> unfold
             -> PatchEmbedding -> PatchTST Encoder -> prediction

The file is self-contained: original PatchTST files are left untouched.
The experiment framework auto-discovers models under models/, so this model
can be launched with  --model PatchTST_Freq  without editing run.py.

First-version decisions (keep simple):
  - per-channel patch_len (v2, no global averaging across channels)
  - discrete rule: period < 16 -> 8; period < 32 -> 16; otherwise -> 32
  - fixed stride = 8
  - per-window variable token count is padded to a fixed maximum and masked
  - optional ratio rule: p = clip(round(period / m), 8, 32), rounded to 4
"""

import torch
from torch import nn

from layers.Embed import PositionalEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer

# Switch to "ratio" later when doing parameter analysis.
PATCH_MODE = "ratio"
PATCH_RATIO = 4.0
PATCH_MIN = 8
PATCH_MAX = 32
PATCH_STEP = 4


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FrequencyAnalyzer(nn.Module):
    """Rule-based FFT analyzer: input [B, C, L] -> (patch_len, period)."""

    def __init__(self, seq_len, mode=PATCH_MODE, p_min=PATCH_MIN,
                 p_max=PATCH_MAX, ratio=PATCH_RATIO, topk=3):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode
        self.p_min = p_min
        self.p_max = p_max
        self.ratio = ratio
        self.topk = topk

    def forward(self, x):
        # x is already instance-normalized: [B, C, L]
        xf = torch.fft.rfft(x, dim=-1)
        energy = xf.abs().pow(2)
        energy[..., 0] = 0.0  # remove DC
        vals, idx = torch.topk(energy, self.topk, dim=-1)
        k = (idx.float() * vals).sum(-1) / vals.sum(-1).clamp(min=1e-6)
        k = k.clamp(min=1.0)  # [B, C]
        period = torch.round(self.seq_len / k).long()  # [B, C]

        if self.mode == "ratio":
            p = ((period.float() / self.ratio / PATCH_STEP).round() * PATCH_STEP)
            p = p.clamp(self.p_min, self.p_max).long()
        else:
            p = torch.where(
                period < 16,
                torch.full_like(period, 8),
                torch.where(
                    period < 32,
                    torch.full_like(period, 16),
                    torch.full_like(period, 32),
                ),
            )
        return p, period


class AdaptivePatchEmbedding(nn.Module):
    """Unfold with per-window patch_len, then pad tokens to a fixed maximum."""

    def __init__(self, d_model, seq_len, stride=8, p_min=PATCH_MIN,
                 p_max=PATCH_MAX, p_step=PATCH_STEP, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.stride = stride
        self.max_num_patches = int((seq_len - p_min) // stride + 1)

        # One projection per possible patch length; no value padding needed.
        self.value_embeddings = nn.ModuleDict({
            str(p): nn.Linear(p, d_model, bias=False)
            for p in range(p_min, p_max + 1, p_step)
        })
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, patch_lens):
        # x: [B, C, L], patch_lens: [B]
        B, C, L = x.shape
        N = self.max_num_patches
        out = x.new_zeros(B, C, N, self.d_model)
        valid = torch.zeros(B, N, dtype=torch.bool, device=x.device)

        # Group samples by patch_len so unfold/linear run on full tensors.
        for p in torch.unique(patch_lens):
            p_item = int(p.item())
            idx = (patch_lens == p).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            x_p = x[idx]                                   # [Bp, C, L]
            patches = x_p.unfold(-1, p_item, self.stride)  # [Bp, C, Np, p]
            n_p = patches.shape[2]
            emb = self.value_embeddings[str(p_item)](patches)  # [Bp, C, Np, D]
            emb_pad = x.new_zeros(idx.numel(), C, N, self.d_model)
            emb_pad[:, :, :n_p] = emb
            out.index_copy_(0, idx, emb_pad)
            valid_group = torch.zeros(
                idx.numel(), N, dtype=torch.bool, device=x.device
            )
            valid_group[:, :n_p] = True
            valid.index_copy_(0, idx, valid_group)

        enc_out = out.reshape(B * C, N, self.d_model)
        enc_out = enc_out + self.position_embedding(enc_out)
        enc_out = self.dropout(enc_out)
        return enc_out, valid


class AttnMaskWrapper:
    """Adapter so FullAttention can consume our additive boolean mask."""

    def __init__(self, mask):
        self.mask = mask


class Model(nn.Module):
    """PatchTST with frequency-guided adaptive patch generation."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.stride = stride

        self.freq_analyzer = FrequencyAnalyzer(self.seq_len)
        self.patch_embedding = AdaptivePatchEmbedding(
            configs.d_model, self.seq_len, stride=stride, dropout=configs.dropout
        )
        self.max_num_patches = self.patch_embedding.max_num_patches

        # mask_flag=True so the additive padding mask is honored; for real
        # tokens the mask only blocks attention to padded positions.
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
            ),
        )

        self.head_nf = configs.d_model * self.max_num_patches
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.pred_len,
                head_dropout=configs.dropout,
            )
        else:
            raise NotImplementedError(
                "PatchTST_Freq v1 supports long/short term forecast only"
            )

    def _build_attn_mask(self, valid):
        # valid: [B, N]. Block valid queries from attending to padded keys.
        # Padded query rows are left unmasked so they stay finite, and their
        # outputs are zeroed after the encoder.
        B, N = valid.shape
        q_valid = valid.unsqueeze(1)          # [B, 1, N]
        k_pad = (~valid).unsqueeze(-1)        # [B, N, 1]
        block = q_valid & k_pad               # [B, N, N]
        return AttnMaskWrapper(block.unsqueeze(1))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # RevIN, same as PatchTST
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)  # [B, C, L]
        B, C, L = x_enc.shape
        patch_lens, periods = self.freq_analyzer(x_enc)  # [B, C]
        enc_out, valid = self.patch_embedding(
            x_enc.reshape(B * C, 1, L), patch_lens.reshape(B * C)
        )

        enc_out, attns = self.encoder(
            enc_out, attn_mask=self._build_attn_mask(valid)
        )
        enc_out = enc_out * valid.unsqueeze(-1)  # zero padded tokens

        enc_out = enc_out.reshape(B, C, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization, same as PatchTST
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise NotImplementedError(
            "PatchTST_Freq v1 supports long/short term forecast only"
        )
