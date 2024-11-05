import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from tqdm import tqdm

from p2p_lending.constants import (
    device,
    learning_rate,
    num_epochs,
    numerical_features,
    prediction_threshold,
    random_state_for_split,
    weight_decay,
    batch_size,
)
from p2p_lending.dataset import Dataset
from p2p_lending.get_data import get_data
from p2p_lending.models.base_model import BaseModel
from p2p_lending.models.deep_feed_forward_model import DeepFeedForwardModel
from p2p_lending.models.logistics_regression_model import LogisticRegressionModel
from p2p_lending.models.transormer_encoder_model import TransformerEncoderModel
from p2p_lending.preprocess_data import preprocess_data
from p2p_lending.utils.dataset import (
    create_dataset_with_embeddings,
    normalize,
    oversample_minority_class,
    split_data,
)
from p2p_lending.utils.loss_attenuation import loss_attenuation
from p2p_lending.utils.mc_dropout import predict_with_mc_dropout

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.debug(f"Using device: {device}")


def main():
    filtered_data = get_data("p2p_lending/data/accepted_2007_to_2018q4.csv")
    processed_data = preprocess_data(filtered_data, percentage_cases=1)

    num_hard_features = (
        processed_data.shape[1] - 2
    )  # -1 because we drop the target column and description column

    models: list[BaseModel] = [
        # TransformerEncoderModel(num_hard_features,2).to(device),
        # LogisticRegressionModel(num_hard_features).to(device),
        DeepFeedForwardModel(num_hard_features, 2).to(device),
        # DeepFeedForwardModel(num_hard_features, 1).to(device),
    ]

    train_data, dev_data, test_data = split_data(processed_data, random_state_for_split)
    train_data = oversample_minority_class(train_data)
    train_data, dev_data, test_data = normalize(
        train_data, dev_data, test_data, numerical_features
    )

    logger.info(f"Creating embeddings for train, dev and test datasets")
    train_dataset_with_embeddings = create_dataset_with_embeddings(train_data, "train")
    dev_dataset_with_embeddings = create_dataset_with_embeddings(dev_data, "dev")
    test_dataset_with_embeddings = create_dataset_with_embeddings(test_data, "test")

    for model in models:
        print("")
        logger.info(f"Training model: {model.__class__.__name__}")
        train(model, train_dataset_with_embeddings, dev_dataset_with_embeddings)

    for model in models:
        (
            auc_test_set,
            gmean_test_set,
            aleatoric_error_correlation,
            epistemic_error_correlation,
        ) = evaluate(model, test_dataset_with_embeddings, check_correlation=True)
        logger.info(
            f"{model.__class__.__name__} - Test set - AUC: {auc_test_set:.4f}, G-mean: {gmean_test_set:.4f}"
        )
        if (
            aleatoric_error_correlation is not None
            and epistemic_error_correlation is not None
        ):
            logger.info(
                f", Aleatoric error correlation: {aleatoric_error_correlation:.4f}, Epistemic error correlation: {epistemic_error_correlation:.4f}"
            )


def train(model: BaseModel, training_dataset: Dataset, dev_dataset: Dataset):
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=weight_decay, lr=learning_rate
    )
    criterion = nn.BCELoss()
    if model.output_dim == 2:
        criterion = loss_attenuation
    best_dev_auc = 0
    best_dev_gmean = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
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

        dev_auc, dev_gmean, aleatoric_error_correlation, epistemic_error_correlation = evaluate(
            model, dev_dataset, check_correlation=True
        )
        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_dev_gmean = dev_gmean
            logger.info("New best dev AUC and G-mean, saving model")
            os.makedirs("p2p_lending/model_weights", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"p2p_lending/model_weights/{model.__class__.__name__}.pth",
            )

        avg_loss = train_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}"
            f": Average Loss: {avg_loss:.4f}"
            f", Dev AUC: {best_dev_auc:.4f}"
            f", Dev G-mean: {best_dev_gmean:.4f}"
            f", Aleatoric error correlation: {aleatoric_error_correlation:.4f}"
            f", Epistemic error correlation: {epistemic_error_correlation:.4f}"
        )
        model.load_state_dict(
            torch.load(f"p2p_lending/model_weights/{model.__class__.__name__}.pth")
        )


def evaluate(model: BaseModel, test_dataset: Dataset, check_correlation: bool = False):
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    model.eval()
    probas = []
    targets = []
    aleatoric_log_variances = []
    epistemic_variances = []

    with torch.no_grad():
        for data, embedding, target in test_loader:
            proba, epistemic_variance, aleatoric_log_variance = predict_with_mc_dropout(
                model, data, embedding
            )
            probas.extend(proba.tolist())
            targets.extend(target.tolist())
            epistemic_variances.extend(epistemic_variance.tolist())
            aleatoric_log_variances.extend(aleatoric_log_variance.tolist())

    probas = torch.tensor(probas)
    epistemic_variances = torch.tensor(epistemic_variances)
    aleatoric_log_variances = torch.tensor(aleatoric_log_variances)
    auc = roc_auc_score(targets, probas)
    predictions = (probas >= prediction_threshold).float()
    sensitivity = recall_score(targets, predictions)
    specificity = recall_score(targets, predictions, pos_label=0)
    gmean = (sensitivity * specificity) ** 0.5

    if model.output_dim != 2 or not check_correlation:
        return auc, gmean, None

    aleatoric_stds = torch.sqrt(torch.exp(aleatoric_log_variances))
    logger.debug(
        f"Aleatoric mean, min & max STD [{aleatoric_stds.mean():.4f}, {aleatoric_stds.min():.4f}, {aleatoric_stds.max():.4f}]"
    )
    epistemic_stds = torch.sqrt(epistemic_variances)
    logger.debug(
        f"Epistemic mean, min & max STD [{epistemic_stds.mean():.4f}, {epistemic_stds.min():.4f}, {epistemic_stds.max():.4f}]"
    )
    error = torch.abs(predictions - torch.tensor(targets))
    aleatoric_error_correlation = torch.corrcoef(
        torch.stack((aleatoric_stds, error))
    )[0, 1]
    epistemic_error_correlation = torch.corrcoef(
        torch.stack((epistemic_stds, error))
    )[0, 1]
    return auc, gmean, aleatoric_error_correlation, epistemic_error_correlation


if __name__ == "__main__":
    main()
