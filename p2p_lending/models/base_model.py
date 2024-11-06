import torch.nn as nn

from p2p_lending.constants import prediction_threshold

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_text_model = False
        self.output_dim = 1
