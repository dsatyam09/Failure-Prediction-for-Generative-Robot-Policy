import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return TransformerModel(cfg, input_dim)


class TransformerModel(BaseModel):
    """
    Causal Transformer encoder for per-timestep failure probability.

    Standard pre-LN transformer with causal mask so the model is "online" — it
    can only attend to past + current timesteps, matching how a real failure
    detector would run during rollout.
    """

    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.n_heads = cfg.model.n_heads
        self.ff_dim = cfg.model.ff_mult * cfg.model.hidden_dim
        self.max_seq_len = cfg.model.max_seq_len

        # Project the (very wide) input features down to hidden_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # Learned positional embeddings (simpler than sinusoidal, fine at this scale)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=cfg.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)

        self._scale_weights(self.cfg.model.init_weight_scale)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        # True at positions that should be MASKED (future). Shape (T, T).
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        B, T, D = x.shape

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert D == self.input_dim, f"Input dim mismatch: {D} != {self.input_dim}"
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"

        # (B, T, hidden_dim)
        h = self.input_proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(positions)

        causal = self._causal_mask(T, x.device)
        h = self.encoder(h, mask=causal, is_causal=True)  # (B, T, hidden_dim)

        h = self.dropout(h)
        p_seq = torch.sigmoid(self.fc(h))  # (B, T, 1)

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
