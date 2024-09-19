import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from logistics_regression_model import LogisticRegressionModel
from preprocessing import preprocess_data

from constants import num_epochs
import torch
import torch.nn as nn

import logging
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


if __name__ == "__main__":
    main()

