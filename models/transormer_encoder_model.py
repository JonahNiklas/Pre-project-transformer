import torch
from models.base_model import BaseModel
from models.transformer_encoder import TransformerEncoder
from constants import transformer_config, embedding_dimension
from torch import nn


class TransformerEncoderModel(BaseModel):
    def __init__(self, hard_features_dim: int):
        super(TransformerEncoderModel, self).__init__()
        self.is_text_model = True
        self.hard_features_dim = hard_features_dim
        
        self.encoder = TransformerEncoder()
        self.fc = DeepFeedForward(hard_features_dim)

    def forward(self, hard_features, embeddings):
        batch_size = hard_features.shape[0]
        assert hard_features.shape == (batch_size, self.hard_features_dim)
        assert embeddings.shape == (
            batch_size,
            transformer_config.max_seq_length,
            embedding_dimension,
        )

        encoded_embeddings = self.encoder(embeddings)
        output = self.fc(torch.cat([hard_features, encoded_embeddings], dim=1))
        assert output.shape == (batch_size, 1)
        return output

class DeepFeedForward(nn.Module):
    def __init__(self, hard_features_dim: int):
        super(DeepFeedForward, self).__init__()
        self.hard_features_dim = hard_features_dim
        self.input_dim = hard_features_dim + embedding_dimension

        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            (self.hard_features_dim + embedding_dimension),
        )
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
