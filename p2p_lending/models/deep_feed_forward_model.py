import torch
import torch.nn as nn

from p2p_lending.constants import deep_feed_forward_hidden_units, dropout_probability
from p2p_lending.models.base_model import BaseModel


class DeepFeedForwardModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DeepFeedForwardModel, self).__init__()
        self.output_dim = output_dim
        self.sequential = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(input_dim, deep_feed_forward_hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(
                deep_feed_forward_hidden_units[0], deep_feed_forward_hidden_units[1]
            ),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(deep_feed_forward_hidden_units[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.sequential(x)
        output = torch.cat(
            [
                torch.clamp(output[:, 0:1], -20, 20),
                torch.clamp(output[:, 1:2], -1e6, 1e6),
            ],
            dim=1,
        )
        assert not torch.isnan(output).any()
        return output
