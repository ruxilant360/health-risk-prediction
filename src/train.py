"""End-to-end training pipeline (refactored from EDA.ipynb + train.ipynb).

Usage:
    python src/train.py --raw_data train.csv
"""

import argparse

from data.preprocess import load_dataset, preprocess, save_dataset
from train import (
    build_model,
    build_test_dataframe,
    compute_sample_weights,
    encode_labels,
    prepare_features_and_target,
    save_artifacts,
    save_test_set,
    split_data,
    train_model,
)


def run_training_pipeline(
    raw_data_path: str,
    processed_data_path: str,
    model_out: str,
    encoder_out: str,
    test_data_out: str,
    test_size: float = 0.2,
    random_state: int = 11,
) -> None:
    print("--Loading raw data--")
    df = load_dataset(raw_data_path)

    print("--Cleaning data--")
    df = preprocess(df)
    save_dataset(df, processed_data_path)

    print("--Preparing features and target--")
    X, y = prepare_features_and_target(df)
    label_encoder, y_encoded = encode_labels(y)

    print("--Splitting train/test--")
    X_train, X_test, y_train, y_test = split_data(
        X, y_encoded, test_size=test_size, random_state=random_state
    )

    print("--Computing class balance sample weights--")
    sample_weights = compute_sample_weights(y_train)

    print("--Training model--")
    model = build_model(random_state=random_state)
    model = train_model(model, X_train, y_train, sample_weight=sample_weights)

    print("--Saving artifacts--")
    save_artifacts(model, label_encoder, model_out, encoder_out)

    df_test = build_test_dataframe(X_test, y_test, label_encoder)
    save_test_set(df_test, test_data_out)

    print("--Done--")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the health-risk classifier")
    parser.add_argument("--raw_data", type=str, default="train.csv", help="Path to raw training CSV")
    parser.add_argument(
        "--processed_data",
        type=str,
        default="final_health_data",
        help="Path to write the cleaned dataset",
    )
    parser.add_argument("--model_out", type=str, default="xgb1.joblib", help="Path to save trained model")
    parser.add_argument(
        "--encoder_out", type=str, default="label_encoder.joblib", help="Path to save label encoder"
    )
    parser.add_argument(
        "--test_data_out", type=str, default="test_data", help="Path to save held-out test set"
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Held-out test set fraction")
    parser.add_argument("--random_state", type=int, default=11, help="Random seed")

    args = parser.parse_args()
    run_training_pipeline(
        raw_data_path=args.raw_data,
        processed_data_path=args.processed_data,
        model_out=args.model_out,
        encoder_out=args.encoder_out,
        test_data_out=args.test_data_out,
        test_size=args.test_size,
        random_state=args.random_state,
    )
