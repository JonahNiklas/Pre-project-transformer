from pydantic import BaseModel

class TransformerConfig(BaseModel):
    d_ff: int
    max_seq_length: int
    dropout: float
    num_heads: int
    activation: str
    num_layers: int