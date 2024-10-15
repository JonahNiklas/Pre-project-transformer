import logging

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from constants import target_column, train_dev_test_split, over_sampling_ratio

from constants import target_column
from dataset import Dataset
from embedding import embed_descriptions

logger = logging.getLogger(__name__)

def create_dataset_with_embeddings(df: pd.DataFrame) -> Dataset:
    embeddings = embed_descriptions(df["desc"])
    df = df.drop(columns=["desc"])
    return Dataset(df, embeddings)

def split_data(
    data: pd.DataFrame, split: tuple[float, float, float] = train_dev_test_split
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=split[2])
    train_data, dev_data = train_test_split(
        train_data, test_size=split[1] / (split[0] + split[1])
    )

    return train_data, dev_data, test_data

def oversample_minority_class(train_data: pd.DataFrame):
    def get_num_cases_per_class(data: pd.DataFrame):
        num_positive_cases = data[data[target_column] == 1].shape[0]
        num_negative_cases = data[data[target_column] == 0].shape[0]
        return num_positive_cases, num_negative_cases

    positive_cases = train_data[train_data[target_column] == 1]
    negative_cases = train_data[train_data[target_column] == 0]
    
    num_positive_cases, num_negative_cases = get_num_cases_per_class(train_data)
    assert num_positive_cases > num_negative_cases
    
    num_negative_samples = num_negative_cases + int((num_positive_cases-num_negative_cases) * (over_sampling_ratio))
    num_positive_samples = num_positive_cases - int((num_positive_cases-num_negative_cases) * (1-over_sampling_ratio))
    logger.debug(
        f"Resampling minority class (negative) from {num_negative_cases} to {num_positive_samples}\n and majority class (positive) from {num_positive_cases} to {num_positive_samples}"
    )
    
    negative_samples = resample(
        negative_cases, replace=True, n_samples=num_negative_samples, random_state=42
    )
    positive_samples = resample(
        positive_cases, replace=False, n_samples=num_positive_samples, random_state=42
    )
    output = pd.concat([positive_samples, negative_samples])
    assert len(output) == num_positive_samples + num_negative_samples
    return output


def normalize(
    train_data: pd.DataFrame, dev_data: pd.DataFrame, test_data: pd.DataFrame, numerical_features: list[str]
):
    scaler = StandardScaler()

    # Separate target column
    train_target = train_data[target_column]
    dev_target = dev_data[target_column]
    test_target = test_data[target_column]

    # Remove target column from features
    train_features = train_data.drop(columns=[target_column])
    dev_features = dev_data.drop(columns=[target_column])
    test_features = test_data.drop(columns=[target_column])

    # Scale numerical features
    train_features_scaled = train_features.copy()
    dev_features_scaled = dev_features.copy()
    test_features_scaled = test_features.copy()
    
    train_features_scaled[numerical_features] = scaler.fit_transform(train_features[numerical_features])
    dev_features_scaled[numerical_features] = scaler.transform(dev_features[numerical_features])
    test_features_scaled[numerical_features] = scaler.transform(test_features[numerical_features])

    # Combine scaled features with unscaled target
    train_data_scaled = pd.concat([train_features_scaled, train_target], axis=1)
    dev_data_scaled = pd.concat([dev_features_scaled, dev_target], axis=1)
    test_data_scaled = pd.concat([test_features_scaled, test_target], axis=1)

    return train_data_scaled, dev_data_scaled, test_data_scaled