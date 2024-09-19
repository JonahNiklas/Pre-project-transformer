import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

from base_model import BaseModel
from constants import num_epochs, target_column, train_dev_test_split
from deep_feed_forward_model import DeepFeedForwardModel
from logistics_regression_model import LogisticRegressionModel
from get_data import get_data
from preprocess_data import preprocess_data

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    filtered_data = get_data("data/accepted_2007_to_2018q4.csv")
    filtered_data = filtered_data.drop(columns=["desc"])
    processed_data = preprocess_data(filtered_data)

    input_dim = processed_data.shape[1] - 1  # -1 because we drop the target column
    models: list[BaseModel] = [
        LogisticRegressionModel(input_dim),
        DeepFeedForwardModel(input_dim),
        # TransformerEncoderModel(),
    ]

    train_data, dev_data, test_data = split_data(processed_data)
    train_data_resampled = oversample_minority_class(train_data)
    train_data_scaled, dev_data_scaled, test_data_scaled = normalize(
        train_data_resampled, dev_data, test_data
    )

    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, train_data_resampled)
        # Evaluate on train_data
        auc_train_set, gmean_train_set = evaluate(model, train_data)
        logger.info(
            f"Train set - AUC: {auc_train_set:.4f}, G-mean: {gmean_train_set:.4f}"
        )
        # Evaluate on dev_data
        auc_dev_set, gmean_dev_set = evaluate(model, dev_data)
        logger.info(f"Dev set - AUC: {auc_dev_set:.4f}, G-mean: {gmean_dev_set:.4f}")


def train(model: nn.Module, data: pd.DataFrame):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(data.drop(columns=[target_column]).values, dtype=torch.float32),
        torch.tensor(data[target_column].values, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def split_data(
    data: pd.DataFrame, split: tuple[float, float, float] = train_dev_test_split
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=split[2])
    train_data, dev_data = train_test_split(
        train_data, test_size=split[1] / (split[0] + split[1])
    )

    return train_data, dev_data, test_data


def evaluate(model: nn.Module, test_data: pd.DataFrame):
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(
            test_data.drop(columns=[target_column]).values, dtype=torch.float32
        ),
        torch.tensor(test_data[target_column].values, dtype=torch.float32),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model.predict(data)
            predictions.extend(output.tolist())
            targets.extend(target.tolist())

    auc = roc_auc_score(targets, predictions)
    sensitivity = recall_score(targets, predictions)
    specificity = recall_score(targets, predictions, pos_label=0)
    gmean = (sensitivity * specificity) ** 0.5

    return auc, gmean


def oversample_minority_class(train_data: pd.DataFrame):
    def get_num_samples_per_class(data: pd.DataFrame):
        num_positive_samples = data[data[target_column] == 1].shape[0]
        num_negative_samples = data[data[target_column] == 0].shape[0]
        return num_positive_samples, num_negative_samples

    num_positive_samples, num_negative_samples = get_num_samples_per_class(train_data)
    assert num_positive_samples > num_negative_samples
    logging.debug(
        f"Resampling minority class (negative) from {num_negative_samples} to {num_positive_samples}"
    )
    positive_samples = train_data[train_data[target_column] == 1]
    negative_samples = train_data[train_data[target_column] == 0]
    negative_samples_upsampled = resample(
        negative_samples, replace=True, n_samples=num_positive_samples, random_state=42
    )
    assert len(negative_samples_upsampled) == num_positive_samples
    oversampled_data = pd.concat([positive_samples, negative_samples_upsampled])
    new_num_positive_samples, new_num_negative_samples = get_num_samples_per_class(
        oversampled_data
    )
    assert new_num_positive_samples == new_num_negative_samples == num_positive_samples
    return oversampled_data


def normalize(
    train_data: pd.DataFrame, dev_data: pd.DataFrame, test_data: pd.DataFrame
):
    scaler = StandardScaler()

    # Separate target column
    train_target = train_data[target_column]
    dev_target = dev_data[target_column]
    test_target = test_data[target_column]

    # Remove target column from features
    train_features = train_data.drop(columns=[target_column])
    dev_features = dev_data.drop(columns=[target_column])
    test_features = test_data.drop(columns=[target_column])

    # Scale features
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )
    dev_features_scaled = pd.DataFrame(
        scaler.transform(dev_features),
        columns=dev_features.columns,
        index=dev_features.index,
    )
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index,
    )

    # Combine scaled features with unscaled target
    train_data_scaled = pd.concat([train_features_scaled, train_target], axis=1)
    dev_data_scaled = pd.concat([dev_features_scaled, dev_target], axis=1)
    test_data_scaled = pd.concat([test_features_scaled, test_target], axis=1)

    return train_data_scaled, dev_data_scaled, test_data_scaled


if __name__ == "__main__":
    main()
