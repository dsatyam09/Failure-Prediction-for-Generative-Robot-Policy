import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return GruModel(cfg, input_dim)


class GruModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.gru = nn.GRU(
            input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            dropout=cfg.model.dropout,
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.n_history_steps = cfg.model.n_history_steps

        self._scale_weights(self.cfg.model.init_weight_scale)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        B, T, D = x.shape
        n = self.n_history_steps

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert D == self.input_dim, f"Input dim mismatch: {D} != {self.input_dim}"

        if n < 0:
            out, _ = self.gru(x)  # (B, T, hidden_dim)
        else:
            x_padded = torch.nn.functional.pad(x, (0, 0, n, 0), mode="constant", value=0)

            x_windows = []
            for t in range(T):
                x_windows.append(x_padded[:, t:t + n, :])

            x_seq = torch.stack(x_windows, dim=1)
            x_seq = x_seq.reshape(B * T, n, D)

            out, _ = self.gru(x_seq)
            out = out[:, -1, :]
            out = out.view(B, T, -1)

        out = self.dropout(out)
        p_seq = torch.sigmoid(self.fc(out))

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
        B, T, D = batch["features"].shape

        scores = self(batch)
        scores = scores.squeeze(-1)

        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)
        time_weights = time_weights.to(scores)

        if self.cfg.model.cumsum:
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)
            seq_loss_fail = time_weights * (- scores)

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
