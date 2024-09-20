import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.is_text_model = False

    def predict(self, x, threshold=0.5):
        proba = self(x)
        return (proba >= threshold).float().squeeze(1)