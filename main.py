import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from logistics_regression_model import LogisticRegressionModel
from preprocessing import preprocess_data

from constants import num_epochs
import torch
import torch.nn as nn

import logging
from sklearn.metrics import roc_auc_score, precision_score, recall_score, fbeta_score
logger = logging.getLogger(__name__)

models = [
    LogisticRegressionModel(),
    # TransformerEncoderModel(),
]

def main():
    logger.info("Reading raw data")
    raw_data = pd.read_csv('data/accepted_2007_to_2018q4.csv')
    logger.info("Preprocessing data")
    filtered_data = preprocess_data(raw_data)
    logger.info("Data preprocessing completed")

    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, filtered_data)

def train(model: nn.Module, data: pd.DataFrame):
    train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in tqdm.range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

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

