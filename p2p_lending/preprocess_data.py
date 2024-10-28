import pandas as pd
import numpy as np
from p2p_lending.constants import categorical_features, percentage_features
import logging

logger = logging.getLogger(__name__)

def preprocess_data(data,percentage_cases=1):
    data = _extract_emp_length(data)
    data = _log_transform_like_paper(data)
    data = _one_hot_encode_features(data, features=categorical_features)
    data = _scale_percentage_features(data, features=percentage_features)
    data = _encode_target(data)
    data = data.sample(frac=percentage_cases)
    return data


def _one_hot_encode_features(data, features):
    return pd.get_dummies(data, columns=features)

def _scale_percentage_features(data, features):
    for feature in features:
        data[feature] = data[feature] / 100
    return data

def _extract_emp_length(data):
    indexes = data[data["emp_length"] == "< 1 year"].index
    data["emp_length"] = data["emp_length"].str.extract("(\d+)").astype(float)
    data.loc[indexes, "emp_length"] = 0.5
    return data


def _log_transform_like_paper(data):
    data["loan_amnt"] = np.log(data["loan_amnt"]).astype(float)
    data["annual_inc"] = np.log(data["annual_inc"]).astype(float)
    return data


def _encode_target(data):
    target_mapping = {"Fully Paid": 1, "Charged Off": 0}
    data["loan_status"] = data["loan_status"].map(target_mapping)
    return data

