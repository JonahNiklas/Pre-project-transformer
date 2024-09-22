import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.is_text_model = False

    def predict(self, hard_feautures, embeddings=None, threshold=0.5):
        if self.is_text_model:
            proba = self(hard_feautures, embeddings)
        else:
            proba = self(hard_feautures)
        return (proba >= threshold).float().squeeze(1)