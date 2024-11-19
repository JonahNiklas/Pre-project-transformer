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
    use_mc_dropout,
    over_sampling_ratio,
    skip_train_load_weights
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
from p2p_lending.utils.mc_dropout import predict_with_mc_dropout, predict

pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Using device: {device}")


def main() -> None:
    filtered_data = get_data("p2p_lending/data/accepted_2007_to_2018q4.csv")
    processed_data = preprocess_data(filtered_data, percentage_cases=1)

    num_hard_features = (
        processed_data.shape[1] - 2
    )  # -2 because we drop the target column and description column

    models: list[BaseModel] = [
        # DeepFeedForwardModel(num_hard_features, 1).to(device),
        DeepFeedForwardModel(num_hard_features, 2).to(device),
        # LogisticRegressionModel(num_hard_features).to(device),
        # TransformerEncoderModel(num_hard_features,1).to(device),
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

    print("")
    logger.info("Evaluating on test set")
    for model in models:
        evaluate(model, test_dataset_with_embeddings, "Test", save_to_file=True)


def train(model: BaseModel, training_dataset: Dataset, dev_dataset: Dataset) -> None:
    if skip_train_load_weights:
        model.load_state_dict(
            torch.load(f"p2p_lending/model_weights/{model.__class__.__name__}.pth")
        )
        evaluate(model, dev_dataset, "Final Devset",save_to_file=True)
        return
    
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=weight_decay, lr=learning_rate
    )
    best_dev_auc: float = 0.0
    best_dev_gmean: float = 0.0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        for i, (data, embedding, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if model.is_text_model:
                output = model(data, embedding)
            else:
                output = model(data)
            loss = model.criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        dev_auc, dev_gmean = evaluate(model, dev_dataset, "Devset")
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
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
        
    model.load_state_dict(
        torch.load(f"p2p_lending/model_weights/{model.__class__.__name__}.pth")
    )
    evaluate(model, dev_dataset, "Final Devset",save_to_file=True)


def evaluate(
    model: BaseModel, dataset: Dataset, dataset_name: str, save_to_file: bool = False
) -> tuple[float, float]:
    probas, targets, epistemic_variances, aleatoric_log_variances = _get_predictions(
        model, dataset
    )
    auc, gmean = _get_auc_and_gmean(probas, targets)
    aleatoric_error_correlation, epistemic_error_correlation = (
        _get_uncertainty_correlation(
            probas, epistemic_variances, aleatoric_log_variances, targets
        )
    )

    logger.info(
        f"{dataset_name} AUC: {auc:.4f}"
        f", {dataset_name} G-mean: {gmean:.4f}"
        f", Aleatoric error correlation: {aleatoric_error_correlation:.4f}"
        f", Epistemic error correlation: {epistemic_error_correlation:.4f}"
    )

    probas_with_high_confidence, targets_with_high_confidence = (
        _get_predictions_with_high_confidence(
            probas, epistemic_variances, aleatoric_log_variances, targets
        )
    )
    if len(probas_with_high_confidence) != 0:
        auc_with_high_confidence, gmean_with_high_confidence = _get_auc_and_gmean(
            probas_with_high_confidence,
            targets_with_high_confidence,
        )
        logger.info(
            f"{dataset_name} AUC with high confidence: {auc_with_high_confidence:.4f}"
            f", {dataset_name} G-mean with high confidence: {gmean_with_high_confidence:.4f}"
        )

    if save_to_file:
        np.save(f'p2p_lending/results/probas_{dataset_name}_{random_state_for_split}_{over_sampling_ratio}.npy', probas)
        np.save(f'p2p_lending/results/targets_{dataset_name}_{random_state_for_split}_{over_sampling_ratio}.npy', targets)
        np.save(f'p2p_lending/results/epistemic_variances_{dataset_name}_{random_state_for_split}_{over_sampling_ratio}.npy', epistemic_variances)
        np.save(f'p2p_lending/results/aleatoric_log_variances_{dataset_name}_{random_state_for_split}_{over_sampling_ratio}.npy', aleatoric_log_variances)
        
    return auc, gmean


def _get_predictions_with_high_confidence(
    probas: torch.Tensor,
    epistemic_variances: torch.Tensor,
    aleatoric_log_variances: torch.Tensor,
    targets: torch.Tensor,
    confidence_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    aleatoric_variances = torch.exp(aleatoric_log_variances)
    total_variance = epistemic_variances + aleatoric_variances
    total_variance_threshold = total_variance.quantile(confidence_threshold)
    high_confidence_indices = total_variance < total_variance_threshold
    return probas[high_confidence_indices], targets[high_confidence_indices]


def _get_auc_and_gmean(
    probas: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    auc = roc_auc_score(targets, probas)
    predictions = (probas >= prediction_threshold).float()
    sensitivity = recall_score(targets, predictions)
    specificity = recall_score(targets, predictions, pos_label=0)
    gmean = (sensitivity * specificity) ** 0.5
    return auc, gmean


def _get_uncertainty_correlation(
    probas: torch.Tensor,
    epistemic_variances: torch.Tensor,
    aleatoric_log_variances: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    # predictions = (probas >= prediction_threshold).float()
    aleatoric_stds = torch.sqrt(torch.exp(aleatoric_log_variances))
    epistemic_stds = torch.sqrt(epistemic_variances)
    logger.debug(
        f"Aleatoric mean, min & max STD [{aleatoric_stds.mean():.4f}, {aleatoric_stds.min():.4f}, {aleatoric_stds.max():.4f}]"
    )
    logger.debug(
        f"Epistemic mean, min & max STD [{epistemic_stds.mean():.4f}, {epistemic_stds.min():.4f}, {epistemic_stds.max():.4f}]"
    )
    error = torch.abs(probas - targets)
    aleatoric_error_correlation = torch.corrcoef(torch.stack((aleatoric_stds, error)))[
        0, 1
    ].item()
    epistemic_error_correlation = torch.corrcoef(torch.stack((epistemic_stds, error)))[
        0, 1
    ].item()
    return aleatoric_error_correlation, epistemic_error_correlation


def _get_predictions(
    model: BaseModel, test_dataset: Dataset
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    model.eval()
    proba_list = []
    targets = []
    aleatoric_log_variances = []
    epistemic_variances = []

    with torch.no_grad():
        for data, embedding, target in test_loader:
            if use_mc_dropout:
                proba, epistemic_variance, aleatoric_log_variance = predict_with_mc_dropout(
                    model, data, embedding
                )
            else:
                proba, aleatoric_log_variance = predict(model, data, embedding)
                epistemic_variance = torch.zeros_like(proba).squeeze()
            proba_list.extend(proba.tolist())
            targets.extend(target.tolist())
            epistemic_variances.extend(epistemic_variance.tolist())
            aleatoric_log_variances.extend(aleatoric_log_variance.tolist())
            
    probas = torch.tensor(proba_list)
    if model.output_dim == 1:
        probas = probas.squeeze(1)

    return (
        probas,
        torch.tensor(targets),
        torch.tensor(epistemic_variances),
        torch.tensor(aleatoric_log_variances),
    )


if __name__ == "__main__":
    main()
