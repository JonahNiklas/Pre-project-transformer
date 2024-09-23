import numpy as np
import pandas as pd
import os
from constants import selected_columns, year_range

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

    filtered_data = _filter_within_date_range(filtered_data)
    filtered_data = _filter_short_descriptions(filtered_data)
    filtered_data = _filter_current_and_late_loans(filtered_data)
    filtered_data = _remove_unnecessary_columns_and_na(filtered_data)
    filtered_data = _dropna(filtered_data)

    _log_description_length_metrics(filtered_data)
    
    logger.info(f"Saving preprocessed data to {save_path}")
    filtered_data.to_parquet(save_path)

    return filtered_data


def _process_dates_and_credit_age(data):
    data["issue_d"] = pd.to_datetime(data["issue_d"], format="%b-%Y")
    data["earliest_cr_line"] = pd.to_datetime(data["earliest_cr_line"], format="%b-%Y")
    data["credit_age"] = (data["issue_d"] - data["earliest_cr_line"]).dt.days / 30
    return data


def _calculate_revolving_income_ratio(data):
    data["total_rev_hi_lim"] = data["total_rev_hi_lim"].fillna(0)
    data["revolving_income_ratio"] = data["total_rev_hi_lim"] / (
        data["annual_inc"] / 12
    )
    return data


def _clean_description(data):
    """Example description:
        Borrower added on 03/18/14 > Looking to consolidate debt as well as purchase new vehicle with loan.<br>
    This function removes the 'Borrower added on [...] >' part and the '<br>' tag from the description.
    """
    data["desc"] = data["desc"].str.replace("<br>", "", regex=False)
    data["desc"] = data["desc"].str.replace("Borrower added on [^ ]+ > ", "", regex=True)
    data["desc"] = data["desc"].str.strip()

    return data


def _add_description_length(data):
    data["desc"] = data["desc"].fillna("").astype(str)
    data["desc_length"] = data["desc"].apply(lambda x: len(tokenize_text(x)))
    return data


def _remove_unnecessary_columns_and_na(data):
    return data.drop(
        columns=["earliest_cr_line", "total_rev_hi_lim", "issue_d"]
    )


def _log_filtered_rows(initial_rows, final_rows, reason):
    rows_removed = initial_rows - final_rows
    percent_removed = (rows_removed / initial_rows) * 100
    logger.debug(
        f"Dropped rows {reason}: {rows_removed}/{initial_rows} ({percent_removed:.2f}%)"
    )


def _filter_within_date_range(data):
    initial_rows = len(data)
    data = data.query(f"issue_d.dt.year >= {year_range[0]} and issue_d.dt.year <= {year_range[1]}")
    final_rows = len(data)
    _log_filtered_rows(initial_rows, final_rows, f"with issue_d outside of {year_range[0]}-{year_range[1]}")
    return data


def _filter_short_descriptions(data):
    initial_rows = len(data)
    filtered_data = data.query("desc_length >= 20")
    final_rows = len(filtered_data)
    _log_filtered_rows(initial_rows, final_rows, "with short descriptions")
    return filtered_data


def _filter_current_and_late_loans(data):
    initial_rows = len(data)
    filtered_data = data[data["loan_status"].isin(["Fully Paid", "Charged Off"])]
    final_rows = len(filtered_data)
    _log_filtered_rows(initial_rows, final_rows, "with current or late loans")
    return filtered_data


def _dropna(data):
    initial_rows = len(data)
    filtered_data = data.dropna()
    final_rows = len(filtered_data)
    _log_filtered_rows(initial_rows, final_rows, "with NA")
    return filtered_data


def _log_description_length_metrics(data):
    logger.debug(f"Description length metrics:\n {data['desc_length'].describe()}")
