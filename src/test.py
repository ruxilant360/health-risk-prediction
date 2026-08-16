"""Model evaluation pipeline (refactored from test_evaluate.ipynb).

Usage:
    python src/test.py --test_data test_data --model_file xgb1.joblib
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from data.preprocess import cast_categorical_columns
from train import TARGET_COL


def load_model_and_encoder(model_path: str, encoder_path: str):
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


def split_features_labels(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop([target_col], axis=1)
    X = cast_categorical_columns(X)
    y = df[target_col]
    return X, y


def run_predictions(model, X: pd.DataFrame):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    return probabilities, predictions


def decode_predictions(predictions, label_encoder) -> np.ndarray:
    return label_encoder.inverse_transform(predictions)


def compute_accuracy(y_true, y_pred) -> float:
    return (np.asarray(y_true) == np.asarray(y_pred)).sum() / len(y_true) * 100


def print_classification_report(y_true, y_pred) -> None:
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str = None) -> None:
    import matplotlib.pyplot as plt

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curves(y_test, raw_scores, label_encoder, output_path: str = None) -> None:
    import matplotlib.pyplot as plt

    classes = list(range(len(label_encoder.classes_)))
    y_test_encoded = label_encoder.transform(y_test)
    y_test_binarized = label_binarize(y_test_encoded, classes=classes)
    class_names = label_encoder.inverse_transform(classes)

    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(class_names):
        true_class = y_test_binarized[:, i]
        score_class = raw_scores[:, i]

        precision, recall, _ = precision_recall_curve(true_class, score_class)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{class_name} (PR-AUC = {pr_auc:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass One-vs-Rest Precision-Recall Curve")
    plt.legend(loc="lower left")

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def run_evaluation(
    test_data_path: str,
    model_path: str,
    encoder_path: str,
    save_plots: bool = False,
) -> None:
    print("--Loading test data--")
    df_test = pd.read_csv(test_data_path)
    X_test, y_test = split_features_labels(df_test)

    print("--Loading model and encoder--")
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)

    print("--Generating predictions--")
    raw_scores, predictions = run_predictions(model, X_test)
    decoded_predictions = decode_predictions(predictions, label_encoder)

    accuracy = compute_accuracy(y_test, decoded_predictions)
    print(f"correct predictions : {accuracy:.2f}%")

    print_classification_report(y_test, decoded_predictions)

    plot_confusion_matrix(
        y_test,
        decoded_predictions,
        labels=model.classes_,
        output_path="confusion_matrix.png" if save_plots else None,
    )
    plot_precision_recall_curves(
        y_test,
        raw_scores,
        label_encoder,
        output_path="precision_recall_curve.png" if save_plots else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the health-risk classifier")
    parser.add_argument("--test_data", type=str, default="test_data", help="Path to held-out test CSV")
    parser.add_argument("--model_file", type=str, default="xgb1.joblib", help="Path to saved model file")
    parser.add_argument(
        "--encoder_file", type=str, default="label_encoder.joblib", help="Path to saved label encoder"
    )
    parser.add_argument(
        "--save_plots", action="store_true", help="Save evaluation plots to PNG instead of displaying them"
    )

    args = parser.parse_args()
    run_evaluation(
        test_data_path=args.test_data,
        model_path=args.model_file,
        encoder_path=args.encoder_file,
        save_plots=args.save_plots,
    )
