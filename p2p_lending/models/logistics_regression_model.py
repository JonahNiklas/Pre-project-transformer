import torch
import torch.nn as nn

from p2p_lending.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, input_dim: int) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.sigmoid(self.linear(x))
        return output
