import logging

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from tqdm import tqdm

from constants import num_epochs, target_column
from dataset import Dataset
from get_data import get_data
from models.base_model import BaseModel
from models.deep_feed_forward_model import DeepFeedForwardModel
from models.logistics_regression_model import LogisticRegressionModel
from models.transormer_encoder_model import TransformerEncoderModel
from preprocess_data import preprocess_data
from utils.dataset import (create_dataset_with_embeddings, normalize,
                           oversample_minority_class, split_data)

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    filtered_data = get_data("data/accepted_2007_to_2018q4.csv")
    processed_data = preprocess_data(filtered_data)

    num_hard_features = (
        processed_data.shape[1] - 2
    )  # -1 because we drop the target column and description column

    models: list[BaseModel] = [
        TransformerEncoderModel(num_hard_features),
        LogisticRegressionModel(num_hard_features),
        DeepFeedForwardModel(num_hard_features),
    ]

    train_data, dev_data, test_data = split_data(processed_data)
    train_data = oversample_minority_class(train_data)
    # train_data, dev_data, test_data = normalize(train_data, dev_data, test_data)

    logger.info(f"Creating embeddings for train, dev and test datasets")
    train_dataset_with_embeddings = create_dataset_with_embeddings(train_data)
    dev_dataset_with_embeddings = create_dataset_with_embeddings(dev_data)
    test_dataset_with_embeddings = create_dataset_with_embeddings(test_data)

    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, train_dataset_with_embeddings)
        # Evaluate on train_data
        auc_train_set, gmean_train_set = evaluate(model, train_dataset_with_embeddings)
        logger.info(
            f"Train set - AUC: {auc_train_set:.4f}, G-mean: {gmean_train_set:.4f}"
        )
        # Evaluate on dev_data
        auc_dev_set, gmean_dev_set = evaluate(model, dev_dataset_with_embeddings)
        logger.info(f"Dev set - AUC: {auc_dev_set:.4f}, G-mean: {gmean_dev_set:.4f}")


def train(model: BaseModel, dataset: Dataset):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        for i, (data, embedding, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if model.is_text_model:
                output = model(data, embedding)
            else:
                output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def evaluate(model: nn.Module, test_dataset: Dataset):
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, embedding, target in test_loader:
            if model.is_text_model:
                output = model.predict(data, embedding)
            else:
                output = model.predict(data)
            predictions.extend(output.tolist())
            targets.extend(target.tolist())

    auc = roc_auc_score(targets, predictions)
    sensitivity = recall_score(targets, predictions)
    specificity = recall_score(targets, predictions, pos_label=0)
    gmean = (sensitivity * specificity) ** 0.5

    return auc, gmean


if __name__ == "__main__":
    main()
