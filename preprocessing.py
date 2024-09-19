import numpy as np
import pandas as pd
import os
from constants import selected_columns

import logging

from utils.bool_string import bool_string
logger = logging.getLogger(__name__)


def preprocess_data(raw_data_path: str, save_path = "data/preprocessed_data.parquet"):
    USE_CACHE = bool_string(os.getenv("USE_CACHE", "true"))
    if USE_CACHE and os.path.exists(save_path):
        logger.info(f"Loading preprocessed data from {save_path}")
        return pd.read_parquet(save_path)

    logger.info(f"Loading raw data from {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path, low_memory=False)
    filtered_data = raw_data[selected_columns]

    filtered_data = _process_dates_and_credit_age(filtered_data)
    filtered_data = _calculate_revolving_income_ratio(filtered_data)
    filtered_data = _filter_short_descriptions(filtered_data)
    filtered_data = _drop_unnecessary_columns_and_na(filtered_data)

    filtered_data = filtered_data.astype(np.float32)

    logger.info(f"Saving preprocessed data to {save_path}")
    filtered_data.to_parquet(save_path)

    return filtered_data


def _process_dates_and_credit_age(data):
    data["issue_d"] = pd.to_datetime(data["issue_d"], format="%b-%Y")
    data["earliest_cr_line"] = pd.to_datetime(data["earliest_cr_line"], format="%b-%Y")
    data = data.query("issue_d <= '2014-12-31'")
    data["credit_age"] = (data["issue_d"] - data["earliest_cr_line"]).dt.days / 30
    return data


def _calculate_revolving_income_ratio(data):
    data["revolving_income_ratio"] = data["total_rev_hi_lim"] / (
        data["annual_inc"] / 12
    )
    return data


def _filter_short_descriptions(data):
    return data.query("desc.str.len() >= 20")


def _drop_unnecessary_columns_and_na(data):
    return data.drop(columns=["earliest_cr_line", "total_rev_hi_lim"]).dropna()

