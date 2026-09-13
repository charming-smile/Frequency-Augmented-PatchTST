"""
FCP-Former style reimplementation (frequency compensation layer).

Paper: FCP-Former: Enhancing Long-Term Multivariate Time Series Forecasting
with Frequency Compensation (Sensors 2025, 25, 5646).

The official implementation is not public, so this module follows the paper's
formulas: FFT on each patch -> learnable complex weights W = W1 + j*W2 with
randomly selected frequency modes -> IFFT back to the time domain, and then
the reconstructed patches go through PatchTST's linear embedding and encoder.
"""

import torch
from torch import nn

from layers.Embed import PositionalEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


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


class FrequencyCompensationLayer(nn.Module):
    """FFT -> learnable complex weighting (with cross-patch mixing) -> IFFT."""

    def __init__(self, patch_len, patch_num, modes=16, init_scale=0.02):
        super().__init__()
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.modes = min(modes, patch_len)
        # W1 + j*W2: per-mode mixing across patches, shape [patch_num, patch_num, patch_len].
        self.W1 = nn.Parameter(
            torch.randn(patch_num, patch_num, patch_len) * init_scale
        )
        self.W2 = nn.Parameter(
            torch.randn(patch_num, patch_num, patch_len) * init_scale
        )
        # Fixed random mode selection for reproducibility (the paper samples M modes).
        idx = torch.randperm(patch_len)[: self.modes]
        self.register_buffer("mode_idx", idx)

    def forward(self, x):  # x: [B, N, P]
        xf = torch.fft.fft(x, dim=-1)  # [B, N, P] complex
        xr = xf.real[:, :, self.mode_idx]  # [B, N, M]
        xi = xf.imag[:, :, self.mode_idx]
        w1 = self.W1[:, :, self.mode_idx]  # [N, N, M]
        w2 = self.W2[:, :, self.mode_idx]

        # Inter-patch frequency fusion: y[b, m, p] = sum_n xf[b, n, p] * w[m, n, p].
        yr = torch.einsum("bnp,mnp->bmp", xr, w1) - torch.einsum(
            "bnp,mnp->bmp", xi, w2
        )
        yi = torch.einsum("bnp,mnp->bmp", xr, w2) + torch.einsum(
            "bnp,mnp->bmp", xi, w1
        )

        yfull = torch.zeros(
            x.shape[0],
            self.patch_num,
            self.patch_len,
            dtype=torch.complex64,
            device=x.device,
        )
        yfull[:, :, self.mode_idx] = torch.complex(yr, yi)
        return torch.fft.ifft(yfull, dim=-1).real


class Model(nn.Module):
    """PatchTST with a frequency compensation layer before linear embedding."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        fcp_modes = getattr(configs, "fcp_modes", 16)
        self.fcl = FrequencyCompensationLayer(patch_len, self.patch_num, modes=fcp_modes)
        self.value_embedding = nn.Linear(patch_len, configs.d_model, bias=False)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
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

        self.head_nf = configs.d_model * self.patch_num
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
            )
        else:
            raise NotImplementedError(
                "PatchTST_FCP supports long/short term forecast only"
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)  # [B, C, L]
        n_vars = x_enc.shape[1]
        x = self.padding_patch_layer(x_enc)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # [B*C, N, P]

        x = self.fcl(x)
        enc_out = self.value_embedding(x)  # [B*C, N, d_model]
        enc_out = enc_out + self.position_embedding(enc_out)
        enc_out = self.dropout(enc_out)

        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

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
            return dec_out[:, -self.pred_len :, :]
        raise NotImplementedError(
            "PatchTST_FCP supports long/short term forecast only"
        )
