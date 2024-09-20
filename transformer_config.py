from pydantic import BaseModel

class TransformerConfig(BaseModel):
    hidden_dim: int
    max_seq_length: int
    dropout: float
    num_heads: int
    activation: str
    num_layers: int