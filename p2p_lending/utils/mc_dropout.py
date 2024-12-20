from typing import Optional
import torch.nn as nn

from p2p_lending.constants import mc_dropout_samples
import torch

import logging

from p2p_lending.models.base_model import BaseModel

logger = logging.getLogger(__name__)


def predict_with_mc_dropout(
    model: BaseModel, data: torch.Tensor, embedding: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _enable_test_time_dropout(model)
    outputs_list: list[torch.Tensor] = []
    for _ in range(mc_dropout_samples):
        if model.is_text_model:
            outputs_list.append(model(data, embedding))
        else:
            outputs_list.append(model(data))
    outputs: torch.Tensor = torch.stack(outputs_list)

    if model.output_dim == 2:
        probas, log_variances = outputs[:, :, 0], outputs[:, :, 1]
    else:
        probas = outputs

    epistemic_variance = probas.var(dim=0)
    probas = probas.mean(dim=0)

    if model.output_dim == 1:
        return probas, epistemic_variance, torch.zeros_like(probas).squeeze()

    aleatoric_log_variance = log_variances.mean(dim=0)

    assert (
        probas.shape
        == aleatoric_log_variance.shape
        == epistemic_variance.shape
        == (len(data),)
    )
    return probas, epistemic_variance, aleatoric_log_variance


def predict(
    model: BaseModel, data: torch.Tensor, embedding: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:

    if model.is_text_model:
        outputs = model(data, embedding)
    else:
        outputs = model(data)

    if model.output_dim == 2:
        probas = outputs[:, 0]
        log_variances = outputs[:, 1]
    else:
        probas = outputs
        log_variances = torch.zeros_like(probas).squeeze()

    return probas, log_variances


def _enable_test_time_dropout(model: BaseModel) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
