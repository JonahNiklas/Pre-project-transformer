import numpy as np
import pandas as pd
import torch

from constants import target_column


class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, hard_features: pd.DataFrame, embeddings: np.ndarray):
        assert len(hard_features) == len(embeddings)
        self.hard_features = torch.tensor(
            hard_features.drop(columns=[target_column]).values, dtype=torch.float32
        )
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.targets = torch.tensor(
            hard_features[target_column].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.hard_features)

    def __getitem__(self, idx):
        return self.hard_features[idx], self.embeddings[idx], self.targets[idx]
