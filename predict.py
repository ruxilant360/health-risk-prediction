import os
import sys
import numpy as np
import pandas as pd
import joblib
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data.preprocess import cast_categorical_columns


def generate_submission(model_path: str, test_path: str, output_path: str, encoder_path: str = "label_encoder.joblib"):
  print("--Loading model from path--")
  model = joblib.load(model_path)

  print("--Loading data from path--")
  df_test = pd.read_csv(test_path)

  #id_col  = df_test["id"]

  X_test  =  df_test.drop(columns =["health_condition"], errors = "ignore")

  X_test = cast_categorical_columns(X_test)

  print("--generating predictions--")
  y_hat = model.predict(X_test)

  le = joblib.load(encoder_path)

  y_hat = le.inverse_transform(y_hat)

  submision  = pd.DataFrame({"Health_Condition": y_hat})

  print(submision.head())

  submision.to_csv(output_path, index = False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description = "Run inference and output predictions"
  )
  parser.add_argument(
    "--test_data",
    type = str,
    default = "script_data",
    help = "Path to input test data CSV"
  )
  parser.add_argument(
    "--model_file",
    type = str,
    default = "xgb1.joblib",
    help = "Path to saved model file"
  )
  parser.add_argument(
    "--encoder_file",
    type = str,
    default = "label_encoder.joblib",
    help = "Path to saved label encoder file"
  )
  parser.add_argument(
      "--output_file",
        type = str,
        default = "submission.csv",
        help = "Path to save submission file"  )


  args = parser.parse_args()
  generate_submission(args.model_file, args.test_data, args.output_file, args.encoder_file)