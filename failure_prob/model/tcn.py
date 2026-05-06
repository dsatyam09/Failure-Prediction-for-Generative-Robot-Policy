import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return TcnModel(cfg, input_dim)


class CausalConv1d(nn.Module):
    """1D conv with left-padding so output[t] depends only on input[<=t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """One block of a Bai/Kolter/Koltun-style TCN: causal conv → ReLU → dropout, twice, with residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.drop(self.act(self.conv1(x)))
        h = self.drop(self.act(self.conv2(h)))
        return self.act(h + residual)


class TcnModel(BaseModel):
    """
    Temporal Convolutional Network (Bai et al., 2018).

    Stack of dilated causal convolutions; receptive field doubles per layer.
    Fully parallel during training (unlike RNNs). For T=50 with kernel=3,
    n_layers=4 gives receptive field 1 + 2*(3-1)*(2^4 - 1) = 61 → covers full sequence.
    """

    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.kernel_size = cfg.model.kernel_size

        layers = []
        in_ch = input_dim
        for i in range(self.n_layers):
            layers.append(TemporalBlock(
                in_ch=in_ch,
                out_ch=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=2 ** i,
                dropout=cfg.model.dropout,
            ))
            in_ch = self.hidden_dim
        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Conv1d(self.hidden_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(cfg.model.dropout)

        self._scale_weights(self.cfg.model.init_weight_scale)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]  # (B, T, D)
        B, T, D = x.shape

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert D == self.input_dim, f"Input dim mismatch: {D} != {self.input_dim}"

        # Conv1d expects (B, C, T)
        h = x.transpose(1, 2)
        h = self.tcn(h)             # (B, hidden_dim, T)
        h = self.dropout(h)
        h = self.fc(h)              # (B, 1, T)
        p_seq = torch.sigmoid(h).transpose(1, 2)  # (B, T, 1)

        if self.cfg.model.cumsum:
            p_seq = cumsum_stopgrad(p_seq, dim=1)
            if self.cfg.model.rmean:
                normalizer = p_seq.new_ones(p_seq.shape).cumsum(dim=1)
                p_seq = p_seq / normalizer

        return p_seq

    def forward_compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]

        scores = self(batch).squeeze(-1)  # (B, T)

        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks).to(scores)

        if self.cfg.model.cumsum:
            seq_loss_success = torch.relu(scores - 0)
            seq_loss_fail = time_weights * (-scores)
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + \
                (success_labels == 0).float()[:, None] * seq_loss_fail
        else:
            criterion = nn.BCELoss(reduction="none")
            if scores.isnan().any():
                import pdb; pdb.set_trace()
            losses = criterion(scores, 1 - success_labels.unsqueeze(-1).expand_as(scores))
            losses[success_labels == 0] *= time_weights[success_labels == 0]

        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses, valid_masks, success_labels, weights,
            self.cfg.model.one_loss_per_seq,
        )

        hard_neg_loss = torch.tensor(0.0).to(scores)
        if self.cfg.model.lambda_hard_heg > 0:
            hard_neg_loss = hard_negative_loss(
                scores, 1 - success_labels, valid_masks,
                self.cfg.model.hard_neg_margin,
                self.cfg.model.hard_neg_beta,
            )
            hard_neg_loss = self.cfg.model.lambda_hard_heg * hard_neg_loss

        monitor_loss += hard_neg_loss

        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
        }

        return monitor_loss, logs
