import numpy as np
import pandas as pd
import os
from constants import selected_columns

import logging

from embedding import tokenize_text
from utils.bool_string import bool_string

logger = logging.getLogger(__name__)


def get_data(
    raw_data_path: str,
    save_path="data/preprocessed_data.parquet",
    use_cache=bool_string(os.getenv("USE_CACHE", "true")),
):
    if use_cache and os.path.exists(save_path):
        logger.info(f"Loading preprocessed data from {save_path}")
        return pd.read_parquet(save_path)

    logger.info(f"Loading raw data from {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path, low_memory=False)
    filtered_data = raw_data[selected_columns]

    filtered_data = _process_dates_and_credit_age(filtered_data)
    filtered_data = _calculate_revolving_income_ratio(filtered_data)
    filtered_data = _clean_description(filtered_data)
    filtered_data = _add_description_length(filtered_data)
    filtered_data = _filter_short_descriptions(filtered_data)
    filtered_data = _drop_unnecessary_columns_and_na(filtered_data)
    filtered_data = _drop_current_and_late_loans(filtered_data)

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


def _clean_description(data):
    """Example description:
        Borrower added on 03/18/14 > Looking to consolidate debt as well as purchase new vehicle with loan.<br>
    This function removes the 'Borrower added on [...] >' part and the '<br>' tag from the description.
    """
    data["desc"] = data["desc"].str.replace("<br>", "")
    data["desc"] = data["desc"].str.replace("Borrower added on [^ ]+ > ", "")
    data["desc"] = data["desc"].str.strip()

    return data


def _add_description_length(data):
    data["desc"] = data["desc"].fillna("").astype(str)
    data["desc_length"] = data["desc"].apply(lambda x: len(tokenize_text(x)))
    return data


def _filter_short_descriptions(data):
    return data.query("desc_length >= 20")


def _drop_unnecessary_columns_and_na(data):
    return data.drop(
        columns=["earliest_cr_line", "total_rev_hi_lim", "issue_d"]
    ).dropna()


def _drop_current_and_late_loans(data):
    return data[data["loan_status"].isin(["Fully Paid", "Charged Off"])]
