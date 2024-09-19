import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import num_epochs, target_column, train_dev_test_split
from logistics_regression_model import LogisticRegressionModel
from preprocessing import preprocess_data

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    filtered_data = preprocess_data("data/accepted_2007_to_2018q4.csv")

    input_dim = filtered_data.shape[1]
    models = [
        LogisticRegressionModel(input_dim),
        # TransformerEncoderModel(),
    ]

    train_data, dev_data, test_data = split_data(filtered_data)

    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, train_data)


def train(model: nn.Module, data: pd.DataFrame):
    # Convert DataFrame to torch Dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(data.drop(columns=[target_column]).values, dtype=torch.float32),
        torch.tensor(data[target_column].values, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        for i, (data, target) in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                logger.error(f"Error in batch {i}: {str(e)}")
                logger.error(f"Data shape: {data.shape}, Target shape: {target.shape}")
                raise

        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def split_data(
    data: pd.DataFrame, split: tuple[float, float, float] = train_dev_test_split
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=split[2])
    train_data, dev_data = train_test_split(train_data, test_size=split[1] / (split[0] + split[1]))
    return train_data, dev_data, test_data

def evaluate(model: nn.Module, test_data: pd.DataFrame):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.tolist())
            targets.extend(target.tolist())

    auc = roc_auc_score(targets, predictions)
    precision = precision_score(targets, [1 if p >= 0.5 else 0 for p in predictions])
    recall = recall_score(targets, [1 if p >= 0.5 else 0 for p in predictions])
    gmean = (precision * recall) ** 0.5

    return auc, gmean

if __name__ == "__main__":
    main()
