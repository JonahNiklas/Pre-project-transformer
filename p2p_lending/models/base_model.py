import torch.nn as nn

from p2p_lending.constants import activation_function
from p2p_lending.utils.loss_attenuation import LossAttenuation

class BaseModel(nn.Module):
    def __init__(self, output_dim = 1) -> None:
        super().__init__()
        self.is_text_model = False
        self.output_dim = output_dim
        if output_dim == 1:
            self.criterion = nn.BCELoss() if activation_function == "sigmoid" else nn.MSELoss()
        else:
            self.criterion = LossAttenuation(nn.BCELoss if activation_function == "sigmoid" else nn.MSELoss)