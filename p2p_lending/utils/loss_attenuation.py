import torch
import torch.nn as nn


class LossAttenuation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, log_variance = output[:, 0], output[:, 1]
        variance = torch.exp(log_variance)
        mse_loss: torch.Tensor = nn.MSELoss(reduction="none")(mean, target.squeeze(1))
        loss = 0.5 * (mse_loss / variance + log_variance)
        return loss.mean()
