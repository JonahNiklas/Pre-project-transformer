import torch
import torch.nn as nn

from p2p_lending.constants import deep_feed_forward_hidden_units, dropout_probability,activation_function
from p2p_lending.models.base_model import BaseModel


class DeepFeedForwardModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DeepFeedForwardModel, self).__init__(output_dim)
        self.sequential = nn.Sequential(
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
        if activation_function == "sigmoid":
            output = torch.cat(
                [
                    torch.sigmoid(output[:, :1]),
                    output[:, 1:],
                ],
                dim=1,
            )
        
        assert not torch.isnan(output).any()
        return output
