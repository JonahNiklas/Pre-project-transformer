import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x, threshold=0.5):
        proba = self.predict_proba(x)
        return (proba >= threshold).float().squeeze(1)

