"""Training logic extracted from train.ipynb."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data.preprocess import cast_categorical_columns

TARGET_COL = "health_condition"


def prepare_features_and_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop([target_col], axis=1)
    X = cast_categorical_columns(X)
    y = df[target_col]
    return X, y


def encode_labels(y: pd.Series):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return le, y_encoded


def split_data(X, y, test_size: float = 0.2, random_state: int = 11):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def compute_sample_weights(y_train) -> np.ndarray:
    return compute_sample_weight("balanced", y_train)


def build_model(random_state: int = 11) -> XGBClassifier:
    return XGBClassifier(random_state=random_state)


def train_model(model: XGBClassifier, X_train, y_train, sample_weight=None) -> XGBClassifier:
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def save_artifacts(model, label_encoder, model_path: str, encoder_path: str) -> None:
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)


def build_test_dataframe(X_test: pd.DataFrame, y_test, label_encoder: LabelEncoder) -> pd.DataFrame:
    """Reassemble a held-out test set (features + decoded labels) for later
    evaluation, mirroring train.ipynb's `df_test` construction."""
    df_test = X_test.copy()
    df_test[TARGET_COL] = label_encoder.inverse_transform(y_test)
    return df_test


def save_test_set(df_test: pd.DataFrame, path: str) -> None:
    df_test.to_csv(path, index=False)
