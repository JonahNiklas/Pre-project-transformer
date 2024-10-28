import torch.nn as nn

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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequential(x)
