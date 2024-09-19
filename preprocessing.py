import pandas as pd
from constants import selected_columns

import logging
logger = logging.getLogger(__name__)


def preprocess_data(raw_data):
    filtered_data = raw_data[selected_columns].copy()

    filtered_data = _process_dates_and_credit_age(filtered_data)
    filtered_data = _calculate_revolving_income_ratio(filtered_data)
    filtered_data = _filter_short_descriptions(filtered_data)
    filtered_data = _drop_unnecessary_columns_and_na(filtered_data)

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

