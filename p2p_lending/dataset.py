import numpy as np
import pandas as pd
import torch

from p2p_lending.constants import target_column
from p2p_lending.constants import device
class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, hard_features: pd.DataFrame, embeddings: np.ndarray) -> None:
        assert len(hard_features) == len(embeddings)
        self.hard_features = torch.tensor(
            hard_features.drop(columns=[target_column]).to_numpy(dtype=np.float32), dtype=torch.float32
        ).to(device)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        self.targets = torch.tensor(
            hard_features[target_column].values, dtype=torch.float32
        ).to(device)

    def __len__(self) -> int:
        return len(self.hard_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.hard_features[idx], self.embeddings[idx], self.targets[idx]
