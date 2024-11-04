import torch.nn as nn

from p2p_lending.constants import mc_dropout_samples
import torch

import logging

logger = logging.getLogger(__name__)


def predict_with_mc_dropout(model, data, embedding=None):
    _enable_test_time_dropout(model)
    outputs = []
    for _ in range(mc_dropout_samples):
        if model.is_text_model:
            outputs.append(model(data, embedding))
        else:
            outputs.append(model(data))
    outputs = torch.stack(outputs)

    if model.output_dim == 2:
        probas, log_variances = outputs[:, :, 0], outputs[:, :, 1]

    epistemic_variance = probas.var(dim=0)
    probas = probas.mean(dim=0)

    if model.output_dim == 1:
        return probas, epistemic_variance, None

    aleatoric_log_variance = log_variances.mean(dim=0)

    assert (
        probas.shape
        == aleatoric_log_variance.shape
        == epistemic_variance.shape
        == (len(data),)
    )
    return probas, epistemic_variance, aleatoric_log_variance


def _enable_test_time_dropout(model):
    counter = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            counter += 1
            module.train()
    logger.debug(f"MC Dropout: Enabled {counter} dropout layers for test time dropout")
