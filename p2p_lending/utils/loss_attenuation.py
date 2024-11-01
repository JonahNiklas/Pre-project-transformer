import torch
import torch.nn as nn

def loss_attenuation(output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        mean, log_variance = output[:, 0], output[:, 1]
        variance = torch.exp(log_variance)  # Convert log variance to variance
        
        # Compute the loss according to the formula
        mse_loss = nn.MSELoss(reduction='none')(mean, target.squeeze(1))  # Compute MSE for each sample
        loss = 0.5 * (mse_loss / variance + log_variance)  # Element-wise operation
        return loss.mean()