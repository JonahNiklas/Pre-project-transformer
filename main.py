import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

from constants import num_epochs, target_column, train_dev_test_split
from logistics_regression_model import LogisticRegressionModel
from get_data import get_data
from pre_process_data import pre_process_data

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    filtered_data = get_data("data/accepted_2007_to_2018q4.csv")
    processed_data = pre_process_data(filtered_data)
    processed_data = processed_data.drop(columns=["desc", "issue_d"])

    input_dim = processed_data.shape[1]
    models = [
        LogisticRegressionModel(input_dim),
        # TransformerEncoderModel(),
    ]

    train_data, dev_data, test_data = split_data(filtered_data)
    train_data_resampled = oversample_minority_class(train_data)
    
    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, train_data_resampled)
        # Evaluate on dev_data
        auc, gmean = evaluate(model, dev_data)
        logger.info(f"Dev set - AUC: {auc:.4f}, G-mean: {gmean:.4f}")


def train(model: nn.Module, data: pd.DataFrame):
    # Convert DataFrame to torch Dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(data.drop(columns=[target_column]).values, dtype=torch.float32),
        torch.tensor(data[target_column].values, dtype=torch.float32),
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
                logger.error(
                    f"Data shape: {data.shape}, Target shape: {target.shape}")
                raise

        avg_loss = train_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def split_data(
    data: pd.DataFrame, split: tuple[float, float, float] = train_dev_test_split
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=split[2])
    train_data, dev_data = train_test_split(
        train_data, test_size=split[1] / (split[0] + split[1])
    )
    return train_data, dev_data, test_data


def evaluate(model: nn.Module, test_data: pd.DataFrame):
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False)
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.tolist())
            targets.extend(target.tolist())

    auc = roc_auc_score(targets, predictions)
    sensitivity = recall_score(
        targets, [1 if p >= 0.5 else 0 for p in predictions])
    specificity = recall_score(
        targets, [1 if p >= 0.5 else 0 for p in predictions], pos_label=0
    )
    gmean = (sensitivity * specificity) ** 0.5

    return auc, gmean


def oversample_minority_class(train_data: pd.DataFrame):
    num_positive_samples = train_data[train_data[target_column] == 1].shape[0]
    num_negative_samples = train_data[train_data[target_column] == 0].shape[0]
    assert num_positive_samples > num_negative_samples
    logging.debug(
        f"Resampling minority class (negative) from {num_negative_samples} to {num_positive_samples}"
    )
    positive_samples = train_data[train_data[target_column] == 1]
    negative_samples = train_data[train_data[target_column] == 0]
    negative_samples_upsampled = resample(
        negative_samples, replace=True, n_samples=num_positive_samples, random_state=42
    )
    return pd.concat([positive_samples, negative_samples_upsampled])

if __name__ == "__main__":
    main()
