import torch
from p2p_lending.models.base_model import BaseModel
from p2p_lending.models.deep_feed_forward_model import DeepFeedForwardModel
from p2p_lending.models.transformer_encoder import TransformerEncoder
from p2p_lending.constants import transformer_config, embedding_dimension


class TransformerEncoderModel(BaseModel):
    def __init__(self, hard_features_dim: int, output_dim: int = 2):
        super(TransformerEncoderModel, self).__init__()
        self.is_text_model = True
        self.hard_features_dim = hard_features_dim
        
        self.encoder = TransformerEncoder()
        self.output_dim = output_dim
        self.fc = DeepFeedForwardModel(hard_features_dim + embedding_dimension,self.output_dim)

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
        assert output.shape == (batch_size, self.output_dim)
        return output