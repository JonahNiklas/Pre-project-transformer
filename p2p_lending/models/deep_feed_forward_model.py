import torch.nn as nn
import torch
from p2p_lending.constants import deep_feed_forward_hidden_units

from p2p_lending.models.base_model import BaseModel


class DeepFeedForwardModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(DeepFeedForwardModel, self).__init__()
        self.output_dim = output_dim
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, deep_feed_forward_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(
                deep_feed_forward_hidden_units[0], deep_feed_forward_hidden_units[1]
            ),
            nn.ReLU(),
            nn.Linear(deep_feed_forward_hidden_units[1], output_dim),
        )

    def forward(self, x):
        output = self.sequential(x)
        return torch.cat(
            [
                torch.sigmoid(output[:, 0:1]),
                output[:, 1:2],
            ],
            dim=1,
        )
