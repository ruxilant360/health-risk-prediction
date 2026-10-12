"""Data loading and cleaning functions extracted from EDA.ipynb / create_dummy_data.ipynb."""

import pandas as pd

# Columns that must be present (non-null) for a row to be usable, per EDA.ipynb.
NA_DROP_SUBSET = [
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

# Columns whose remaining missing values are imputed with the column mode.
MODE_FILL_COLUMNS = ["sleep_duration", "stress_level"]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["id"], errors="ignore")


def drop_missing_rows(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    return df.dropna(subset=subset)


def fill_missing_with_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
    return df


def cast_categorical_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Cast object-dtype columns to pandas 'category' dtype (needed for XGBoost's
    native categorical support)."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns
    df[columns] = df[columns].astype("category")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline from EDA.ipynb: drop id, drop rows missing key
    fields, then impute the remaining sparse columns with their mode."""
    df = drop_id_column(df)
    df = drop_missing_rows(df, NA_DROP_SUBSET)
    df = fill_missing_with_mode(df, MODE_FILL_COLUMNS)
    return df


def create_dummy_sample(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Take the first n rows of a dataset, e.g. to build a small smoke-test
    payload for predict.py (see create_dummy_data.ipynb)."""
    return df.iloc[:n].copy()
