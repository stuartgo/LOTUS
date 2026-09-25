"""
Supervised event-localisation model for day-long accelerometer recordings.

Pipeline
    (N, T, C) raw signal
      -> PatchEmbedding     : non-overlapping conv patches -> (N, P, d_model)
      -> [PositionalEncoding, optional]
      -> temporal trunk     : TCN or pre-norm transformer over the P patches
      -> Linear head        : one logit per patch (distribution head)
                              or one scalar per recording (regression head)

For the distribution head the target is a Gaussian-smoothed distribution over
patch positions centred on the labelled event time, trained with soft-label
cross-entropy and decoded as the expected position under the softmax.

Input shape : (N, T, C) -- batch, time steps, channels
Labels      : (N,) event time in clock hours (e.g. 23.5 == 23:30)
"""

from __future__ import annotations

import math
from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN
from torch import Tensor

from transformer_encoder import TransformerEncoder


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences of shape (N, T, d_model)."""

    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )                                                                # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)                                   # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x : (N, T, d_model)"""
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Patch embedding (non-overlapping patches + conv feature extractor)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Split (N, T, C) into non-overlapping patches of ``patch_len`` steps and
    map each patch to a ``d_model`` vector with a strided conv stack.

    A conv stack picks up movement structure inside a patch that a single
    ``Linear(patch_len * C, d_model)`` cannot. ``d_model`` must be divisible by
    8 (GroupNorm with 4 groups on ``d_model // 2`` channels).
    """

    def __init__(self, in_channels: int, d_model: int, patch_len: int):
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError(f"d_model must be divisible by 8, got {d_model}")
        self.patch_len = patch_len
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(4, d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(4, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x : (N, T, C) -> (N, n_patches, d_model). A trailing partial patch is dropped."""
        N = x.shape[0]
        x = x.unfold(1, self.patch_len, self.patch_len)   # (N, P, C, patch_len)
        x = x.flatten(0, 1)                               # (N*P, C, patch_len)
        x = self.proj(x).squeeze(-1)                      # (N*P, d_model)
        return x.view(N, -1, x.shape[-1])                 # (N, P, d_model)


# ---------------------------------------------------------------------------
# Transformer trunk (drop-in alternative to the TCN)
# ---------------------------------------------------------------------------

class TransformerTrunk(nn.Module):
    """Pre-norm transformer encoder over the patch sequence, (N, P, d) -> (N, P, d).

    The encoder blocks are pre-norm, so a final LayerNorm is applied here.
    Same in/out contract as ``TCN(input_shape="NLC")``.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            input_dim=d_model,
            num_heads=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.encoder(x))


# ---------------------------------------------------------------------------
# Lightning model
# ---------------------------------------------------------------------------

class SleepModel(pl.LightningModule):
    """Supervised patch-wise event-localisation model.

    Parameters
    ----------
    in_channels         : C, input channels of the raw time series.
    d_model             : embedding dimension (divisible by 8).
    temporal_model      : "tcn" or "transformer".
    lr, weight_decay    : AdamW settings (cosine-annealed over max_epochs).
    dropout             : dropout used in the positional encoding and transformer.
    max_len             : max number of patches for the positional encoding.
    patch_len           : time steps per non-overlapping patch.
    sample_rate         : samples per second of the input signal
                          (10 for 10 Hz, 1 for 1 Hz, 1/60 for 1-min epochs).
    start_hour          : clock hour at which each recording window starts.
    label_sigma_min     : width (minutes) of the Gaussian soft label.
    seq_len             : recording length in samples.
    num_channels        : TCN channel widths, one per level (TCN only).
    kernel_size_tcn     : TCN kernel size (TCN only).
    nhead               : attention heads (transformer only).
    transformer_layers  : encoder blocks (transformer only).
    transformer_ff_dim  : feed-forward width (transformer only).
    head_type           : "distribution" -- one logit per patch, decoded by
                          ``calc_pred``; or "regression" -- one scalar per
                          recording from a mean-pool over patches. The
                          regression head is the ablation arm: it cannot
                          represent more than one candidate event, so on
                          multimodal nights it falls between the modes.
    regression_loss     : "l1" (conditional median) or "l2" (conditional mean).
    k                   : half-width, in patches, of the window used by
                          ``calc_pred_local``.
    use_pos_enc         : add sinusoidal positional encoding before the trunk.
                          Off in the published configuration.
    startup_epochs, label_sigma_min_wide :
                          unused; kept so checkpoints saved with these
                          hyperparameters still load.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        temporal_model: str = "tcn",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        max_len: int = 5_000,
        patch_len: int = 1,
        startup_epochs: int = 10,
        sample_rate: float = 10,
        start_hour: float = 15,
        label_sigma_min: float = 5,
        label_sigma_min_wide: float = 10,
        seq_len: int = 3000,
        num_channels: Sequence[int] = (64,),
        kernel_size_tcn: int = 3,
        nhead: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        head_type: str = "distribution",
        regression_loss: str = "l1",
        k: int = 1,
        use_pos_enc: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if temporal_model not in ("tcn", "transformer"):
            raise ValueError(f"temporal_model must be 'tcn' or 'transformer', got {temporal_model!r}")
        if head_type not in ("distribution", "regression"):
            raise ValueError(f"head_type must be 'distribution' or 'regression', got {head_type!r}")
        if regression_loss not in ("l1", "l2"):
            raise ValueError(f"regression_loss must be 'l1' or 'l2', got {regression_loss!r}")

        self.k = k
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len

        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_len)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        if temporal_model == "tcn":
            num_channels = list(num_channels)
            self.temporal_model = TCN(
                num_inputs=d_model,
                kernel_size=kernel_size_tcn,
                num_channels=num_channels,
                input_shape="NLC",
                causal=False,
            )
            trunk_dim = num_channels[-1]
        else:
            self.temporal_model = TransformerTrunk(
                d_model=d_model,
                nhead=nhead,
                num_layers=transformer_layers,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
            )
            trunk_dim = d_model

        self.head = nn.Linear(trunk_dim, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """(N, T, C) raw signal -> head output.

        distribution : (N, n_patches) logits, one per patch.
        regression   : (N,) event position as a fraction of the recording.
        """
        x = self.patch_embed(x)                  # (N, P, d_model)
        if self.hparams.use_pos_enc:
            x = self.pos_enc(x)
        emb = self.temporal_model(x)             # (N, P, trunk_dim)
        if self.hparams.head_type == "regression":
            return self.head(emb.mean(dim=1)).squeeze(-1)   # (N,)
        return self.head(emb).squeeze(-1)                   # (N, P)

    # ------------------------------------------------------------------
    # Targets and decoding
    # ------------------------------------------------------------------

    def hours_to_sample(self, hours: Tensor) -> Tensor:
        """Clock hours -> position within the recording, in samples."""
        hp = self.hparams
        return ((hours - hp.start_hour) % 24) * 3600 * hp.sample_rate

    def sample_to_hours(self, sample: Tensor) -> Tensor:
        """Position within the recording, in samples -> clock hours."""
        hp = self.hparams
        return (sample / (3600 * hp.sample_rate) + hp.start_hour) % 24

    def _soft_targets(self, hours: Tensor) -> tuple[Tensor, Tensor]:
        """Gaussian-smoothed single-event targets over patch positions.

        Args:
            hours : (N,) event time in clock hours.

        Returns:
            target_soft  : (N, n_patches) normalised Gaussian centred on the
                           event's continuous patch position.
            event_sample : (N,) event position within the recording, in samples.
        """
        hp = self.hparams
        step = self.patch_len
        n_patches = self.n_patches

        event_sample = self.hours_to_sample(hours)                              # (N,)
        # Patch i is centred at i*step + patch_len/2.
        event_pos = ((event_sample - step / 2) / step).clamp(0, n_patches - 1)
        event_pos = event_pos.unsqueeze(1)                                      # (N, 1)
        positions = torch.arange(n_patches, device=hours.device,
                                 dtype=torch.float).unsqueeze(0)                # (1, P)
        sigma = hp.label_sigma_min * 60 * hp.sample_rate / step                # minutes -> patches
        target_soft = torch.softmax(-((positions - event_pos) ** 2) / (2 * sigma ** 2), dim=1)
        return target_soft, event_sample

    def calc_pred(self, logits: Tensor, n_patches: int | None = None) -> Tensor:
        """Expected event position under softmax(logits), in samples. (N, P) -> (N,)

        ``n_patches`` is ignored (inferred from ``logits``); kept for older call sites.
        """
        probs = torch.softmax(logits, dim=1)
        positions = torch.arange(logits.shape[1], device=logits.device).float()
        pred_patch = (probs * positions).sum(dim=1)
        return pred_patch * self.patch_len + self.patch_len / 2

    def calc_pred_local(self, logits: Tensor) -> Tensor:
        """Alternative decoder: centre of mass within +-k patches of the mode.

        Robust to a second, smaller mode pulling the expectation between them.
        """
        probs = torch.softmax(logits, dim=1)
        peak = probs.argmax(dim=1)
        pos = torch.arange(logits.shape[1], device=logits.device).float()
        window = (pos.unsqueeze(0) - peak.unsqueeze(1)).abs() <= self.k
        w = probs * window
        pred_patch = (w * pos).sum(1) / w.sum(1).clamp_min(1e-8)
        return pred_patch * self.patch_len + self.patch_len / 2

    @torch.no_grad()
    def predict_hours(self, x: Tensor) -> Tensor:
        """Convenience: (N, T, C) -> (N,) predicted event time in clock hours."""
        out = self(x)
        if self.hparams.head_type == "regression":
            sample = out.clamp(0.0, 1.0) * self.hparams.seq_len
        else:
            sample = self.calc_pred(out)
        return self.sample_to_hours(sample)

    # ------------------------------------------------------------------
    # Supervised objective
    # ------------------------------------------------------------------

    def do_supervised(self, out: Tensor, y: Tensor, stage: str | None = None):
        """Localisation loss for the active head.

        Args:
            out : head output from ``forward``.
            y   : (N,) event time in clock hours.
            stage : unused; kept for older call sites.

        Returns:
            loss        : scalar loss.
            loc_err_min : (N,) absolute localisation error, minutes.
            pred_sample : (N,) predicted event position, samples.

        Both heads are scored with the same ``loc_err_min``, so they are
        directly comparable.
        """
        hp = self.hparams
        target_soft, event_sample = self._soft_targets(y)

        if hp.head_type == "regression":
            # Normalised to [0, 1] so the target scale is independent of
            # seq_len, sample_rate and patch_len.
            target_norm = event_sample / hp.seq_len
            loss = (F.l1_loss(out, target_norm) if hp.regression_loss == "l1"
                    else F.mse_loss(out, target_norm))
            pred_sample = out.clamp(0.0, 1.0) * hp.seq_len
        else:
            loss = F.cross_entropy(out, target_soft)
            pred_sample = self.calc_pred(out)

        loc_err_min = (pred_sample - event_sample).abs() / (60 * hp.sample_rate)
        return loss, loc_err_min, pred_sample

    # ------------------------------------------------------------------
    # Augmentation (not used by the default training step)
    # ------------------------------------------------------------------

    def shift_data_randomly(self, x: Tensor, y: Tensor):
        """Circularly roll each sample by a random whole number of patches and
        shift its label by the same amount.

        Expects the *patched* tensor (N, P, d_model), so shifts are quantised to
        whole patches. torch.roll cannot shift rows by different amounts, so a
        per-row gather index is built instead.
        """
        N, T, D = x.shape
        shift_patches = torch.randint(T, (N,), device=x.device)
        shift_hours = shift_patches.float() * self.patch_len / (3600 * self.hparams.sample_rate)

        t_idx = torch.arange(T, device=x.device).unsqueeze(0)          # (1, T)
        src = (t_idx - shift_patches.unsqueeze(1)) % T                 # (N, T)
        x = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, D))

        y = (y + shift_hours.to(y.dtype)) % 24
        return x, y

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str) -> Tensor:
        x, y, _pids, _days = batch
        out = self(x)
        loss, loc_err_min, _ = self.do_supervised(out, y)

        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loc_err_mean", loc_err_min.mean(),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loc_err_median", loc_err_min.median(),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        t_max = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
