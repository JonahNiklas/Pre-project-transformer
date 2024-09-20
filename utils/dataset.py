import logging

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from constants import num_epochs, target_column, train_dev_test_split

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
    def get_num_samples_per_class(data: pd.DataFrame):
        num_positive_samples = data[data[target_column] == 1].shape[0]
        num_negative_samples = data[data[target_column] == 0].shape[0]
        return num_positive_samples, num_negative_samples

    num_positive_samples, num_negative_samples = get_num_samples_per_class(train_data)
    assert num_positive_samples > num_negative_samples
    logger.debug(
        f"Resampling minority class (negative) from {num_negative_samples} to {num_positive_samples}"
    )
    positive_samples = train_data[train_data[target_column] == 1]
    negative_samples = train_data[train_data[target_column] == 0]
    negative_samples_upsampled = resample(
        negative_samples, replace=True, n_samples=num_positive_samples, random_state=42
    )
    assert len(negative_samples_upsampled) == num_positive_samples
    oversampled_data = pd.concat([positive_samples, negative_samples_upsampled])
    new_num_positive_samples, new_num_negative_samples = get_num_samples_per_class(
        oversampled_data
    )
    assert new_num_positive_samples == new_num_negative_samples == num_positive_samples
    return oversampled_data


def normalize(
    train_data: pd.DataFrame, dev_data: pd.DataFrame, test_data: pd.DataFrame
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

    # Scale features
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )
    dev_features_scaled = pd.DataFrame(
        scaler.transform(dev_features),
        columns=dev_features.columns,
        index=dev_features.index,
    )
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index,
    )

    # Combine scaled features with unscaled target
    train_data_scaled = pd.concat([train_features_scaled, train_target], axis=1)
    dev_data_scaled = pd.concat([dev_features_scaled, dev_target], axis=1)
    test_data_scaled = pd.concat([test_features_scaled, test_target], axis=1)

    return train_data_scaled, dev_data_scaled, test_data_scaled