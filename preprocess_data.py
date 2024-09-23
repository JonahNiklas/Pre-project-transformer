import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from constants import categorical_features
import logging

logger = logging.getLogger(__name__)

def preprocess_data(data):
    data = _extract_emp_length(data)
    data = _grade_to_numeric(data)
    data = _log_transform_like_paper(data)
    data = _one_hot_encode_features(data, features=categorical_features)
    data = _encode_target(data)

    return data


def _one_hot_encode_features(data, features):
    return pd.get_dummies(data, columns=features)


def _grade_to_numeric(data):
    grade_mapping = {"G": 0, "F": 1, "E": 2, "D": 3, "C": 4, "B": 5, "A": 6}
    data["grade"] = data["grade"].map(grade_mapping) / 6
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

