import torch.nn as nn

class BaseModel(nn.Module):
    def predict(self, x, threshold=0.5):
        proba = self(x)
        return (proba >= threshold).float().squeeze(1)