from torch import nn
import torch
import math
from constants import transformer_config as config, embedding_dimension

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.max_seq_length = config.max_seq_length

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_dimension, config.dropout, config.max_seq_length
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.num_layers,
            # norm=nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, config.max_seq_length, embedding_dimension)
        x = x.transpose(0, 1)  
        assert x.shape == (config.max_seq_length, batch_size, embedding_dimension)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=config.max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
