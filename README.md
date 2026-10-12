# Health Risk Classifier

An end-to-end machine learning pipeline that sorts people into **fit**, **at-risk** or **unhealthy** from lifestyle and biometric data such as sleep, heart rate, BMI, activity, diet and stress. It is built on XGBoost and covers every stage from exploratory analysis to command-line inference.

**Results:** **90.7% accuracy** on a held-out test set of 89,246 records, with **≥ 0.90 recall on every class**, despite a heavily imbalanced target (86% / 8% / 6%).

---

## Highlights

- **Large real-world dataset:** about 690K raw records were cleaned down to 446K usable rows.
- **Principled missing-data handling:** rows missing key fields are dropped, and the sparse columns (`sleep_duration`, `stress_level`) are filled with mode imputation.
- **Class-imbalance aware:** balanced sample weights stop the model from defaulting to the majority class, so the minority classes keep recall above 0.9.
- **Native categorical support:** XGBoost reads pandas `category` dtypes directly, with no one-hot explosion.
- **From notebooks to production code:** the exploratory notebooks were refactored into modular, testable Python packages with CLI entry points for training, evaluation and inference.
- **Reproducible:** a stratified train/test split, fixed random seeds, and a model and label encoder serialised with `joblib`.

---

## Model Performance

Evaluated on a stratified 20% hold-out set (89,246 samples):

| Class         | Precision | Recall | F1-score | Support |
|---------------|:---------:|:------:|:--------:|--------:|
| at-risk       | 0.99      | 0.90   | 0.94     | 76,589  |
| fit           | 0.59      | 0.92   | 0.72     | 5,103   |
| unhealthy     | 0.63      | 0.91   | 0.74     | 7,554   |
| **Accuracy**  |           |        | **0.91** | 89,246  |
| Macro avg     | 0.74      | 0.91   | 0.80     |         |
| Weighted avg  | 0.93      | 0.91   | 0.91     |         |

The model was tuned to favour **recall** on the minority classes. In a health-screening setting, missing an unhealthy individual costs more than a false alarm.

---

## Features

| Feature                   | Type        | Description                              |
|---------------------------|-------------|------------------------------------------|
| `sleep_duration`          | numeric     | Hours of sleep per night                 |
| `heart_rate`              | numeric     | Resting heart rate (bpm)                 |
| `bmi`                     | numeric     | Body mass index                          |
| `calorie_expenditure`     | numeric     | Daily calories burned                    |
| `step_count`              | numeric     | Daily steps                              |
| `exercise_duration`       | numeric     | Minutes of exercise per day              |
| `water_intake`            | numeric     | Litres of water per day                  |
| `diet_type`               | categorical | e.g. `veg`, `non-veg`                    |
| `stress_level`            | categorical | `low` / `medium` / `high`                |
| `sleep_quality`           | categorical | e.g. `poor`, `average`                   |
| `physical_activity_level` | categorical | e.g. `sedentary`, `moderate`, `active`   |
| `smoking_alcohol`         | categorical | e.g. `yes`, `occasional`                 |
| `gender`                  | categorical | `male` / `female` / `other`              |

**Target:** `health_condition`, one of `fit`, `at-risk` or `unhealthy`.

---

## Project Structure

```
health-risk-project/
├── assets/
│   ├── xgb1.joblib              # Trained XGBoost model
│   └── label_encoder.joblib     # Fitted label encoder
├── notebooks/
│   ├── EDA.ipynb                # Exploratory analysis & cleaning
│   ├── train.ipynb              # Model training experiments
│   ├── test_evaluate.ipynb      # Evaluation, confusion matrix, PR curves
│   └── create_dummy_data.ipynb  # Builds a small sample for inference
├── src/
│   ├── data/
│   │   └── preprocess.py        # Loading, cleaning, imputation, dtype casting
│   ├── train/
│   │   └── core.py              # Encoding, splitting, weighting, model build/fit
│   ├── train.py                 # End-to-end training CLI
│   └── test.py                  # Evaluation CLI
└── predict.py                   # Inference CLI
```

---

## Getting Started

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd health-risk-project

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

### 2. Run inference with the pre-trained model

```bash
python predict.py \
  --test_data path/to/input.csv \
  --model_file assets/xgb1.joblib \
  --encoder_file assets/label_encoder.joblib \
  --output_file submission.csv
```

The input CSV needs the 13 feature columns listed above. A `health_condition` column is ignored if it is present. Predictions are written to `submission.csv`:

```
Health_Condition
at-risk
fit
...
```

### 3. Train from scratch

```bash
python src/train.py \
  --raw_data path/to/train.csv \
  --processed_data data/final_health_data \
  --model_out assets/xgb1.joblib \
  --encoder_out assets/label_encoder.joblib \
  --test_data_out data/test_data
```

This command cleans the raw data, makes a stratified 80/20 split, trains a class-weighted XGBoost model, and saves the model, the encoder and the held-out test set.

### 4. Evaluate

```bash
python src/test.py \
  --test_data data/test_data \
  --model_file assets/xgb1.joblib \
  --encoder_file assets/label_encoder.joblib \
  --save_plots
```

It prints accuracy and a full classification report. It also saves `confusion_matrix.png` and `precision_recall_curve.png`, which plot one-vs-rest PR-AUC for each class.

---

## Methodology

1. **EDA:** profiled the 690K rows. Found up to 12% missingness in some columns and a strongly imbalanced target. Examined distributions, outliers (box plots) and feature correlations (heatmap).
2. **Cleaning:** dropped the `id` column and the rows missing critical fields. Mode-imputed `sleep_duration` and `stress_level`, the two columns with the most missing values.
3. **Modelling:** label-encoded the target and made a stratified train/test split. Trained an `XGBClassifier` on native categorical features with `compute_sample_weight("balanced")` to counter class imbalance.
4. **Evaluation:** measured accuracy, per-class precision/recall/F1, a confusion matrix and one-vs-rest precision-recall curves.
5. **Productionising:** refactored the notebook logic into reusable modules and argparse-driven scripts for training, evaluation and inference.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · joblib · Matplotlib · seaborn · Jupyter
