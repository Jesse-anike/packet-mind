# PACKET MIND — Network Traffic ML Classifier

Train and compare Decision Tree and Random Forest classifiers on an authorized, labeled network-flow CSV export. The workflow is suitable for carefully prepared CIC-IDS2017 subsets and other numeric flow datasets.

## Features
- Reads a labeled CSV and validates feature/label columns.
- Keeps numeric, finite feature values and drops incomplete rows.
- Performs a reproducible train/test split, using stratification where possible.
- Compares Decision Tree and Random Forest models.
- Reports accuracy, per-class precision/recall/F1, confusion matrices, and class distribution.
- Optional XGBoost comparison when the `xgboost` package is installed.

## Requirements
- Python 3.10 or later
- scikit-learn (see `requirements.txt`)
- Optional: XGBoost for the third model

Install dependencies:
```bash
python -m pip install -r requirements.txt
# Optional XGBoost model
python -m pip install xgboost
```

## Run
```bash
python main.py sample.csv --label-column Label
python main.py labeled_flows.csv --label-column Label --include-xgboost
```

The sample dataset is synthetic and exists only to exercise the pipeline; do not present its metrics as real-world intrusion-detection performance. For real experiments, prepare an authorized and documented CIC-IDS2017 subset, review its labels, remove leakage-prone identifiers, and record the preprocessing decisions.

## Validate
```bash
python -m unittest discover -s tests -v
python main.py --help
```

## Evaluation cautions
Accuracy alone can hide poor minority-class detection. Inspect class balance, precision, recall, F1, confusion matrices, false positives, false negatives, data leakage, duplicate flows, and dataset representativeness before drawing conclusions.

## Safe use
Use only authorized network datasets and exports. This project is offline analysis: it does not capture network traffic, scan hosts, or block connections.

## Portfolio context
Built for Anike-Moses Melie Jesse's cybersecurity and cloud-security portfolio to demonstrate data preparation, supervised classification, reproducible model evaluation, and careful interpretation of network-security telemetry.
