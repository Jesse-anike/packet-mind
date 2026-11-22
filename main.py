from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import sys
from pathlib import Path


def parse_dataset(text: str, label_column: str = "Label"):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        return {"error": "CSV header is missing."}
    reader.fieldnames = [name.strip() if name else name for name in reader.fieldnames]
    if label_column not in reader.fieldnames:
        return {"error": f"Label column '{label_column}' was not found.", "available_columns": reader.fieldnames}

    feature_columns = [name for name in reader.fieldnames if name != label_column]
    features, labels = [], []
    skipped = 0
    for row in reader:
        vector = []
        try:
            for col in feature_columns:
                raw = row.get(col)
                if raw is None or not raw.strip():
                    raise ValueError("missing feature")
                number = float(raw.strip())
                if not math.isfinite(number):
                    raise ValueError("non-finite feature")
                vector.append(number)
            label = (row.get(label_column) or "").strip()
            if not label:
                raise ValueError("missing label")
        except (ValueError, AttributeError):
            skipped += 1
            continue
        features.append(vector)
        labels.append(label)

    return {
        "features": features,
        "labels": labels,
        "feature_columns": feature_columns,
        "skipped_rows": skipped,
    }


def evaluate_models(features, labels, include_xgboost: bool = False):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return {"error": "scikit-learn is required. Install it with: python -m pip install -r requirements.txt"}

    class_counts = collections.Counter(labels)
    if len(features) < 10 or len(class_counts) < 2:
        return {
            "error": "Need at least 10 complete numeric rows and at least 2 label classes.",
            "rows_usable": len(features),
            "classes": dict(class_counts),
        }

    test_count = max(2, math.ceil(len(features) * 0.25))
    can_stratify = min(class_counts.values()) >= 2 and test_count >= len(class_counts)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels if can_stratify else None,
    )

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            models["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob" if len(class_counts) > 2 else "binary:logistic",
                eval_metric="mlogloss" if len(class_counts) > 2 else "logloss",
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            return {"error": "XGBoost was requested but is not installed. Run: python -m pip install xgboost"}

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classes": [str(item) for item in model.classes_],
        }

    return {
        "rows_used": len(features),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_distribution": dict(class_counts),
        "models": results,
        "evaluation_note": (
            "Metrics are meaningful only when the dataset is representative, labels are trustworthy, "
            "duplicate/leakage risks are controlled, and minority-class recall is reviewed."
        ),
    }


def analyze(text: str, label_column: str = "Label", include_xgboost: bool = False):
    parsed = parse_dataset(text, label_column=label_column)
    if "error" in parsed:
        return parsed
    report = evaluate_models(parsed["features"], parsed["labels"], include_xgboost=include_xgboost)
    if "error" in report:
        report["skipped_rows"] = parsed["skipped_rows"]
        return report
    report["feature_columns"] = parsed["feature_columns"]
    report["skipped_rows"] = parsed["skipped_rows"]
    report["label_column"] = label_column
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare Decision Tree and Random Forest classifiers on authorized labeled network-flow CSV data."
    )
    parser.add_argument("target", nargs="?", default="-", help="CSV path; use - or omit to read stdin")
    parser.add_argument("--label-column", default="Label", help="Name of the target/label column")
    parser.add_argument("--include-xgboost", action="store_true", help="Also evaluate XGBoost if installed")
    parser.add_argument("--output", help="Optional JSON report output path")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.target == "-" else Path(args.target).read_text(encoding="utf-8", errors="replace")
    report = analyze(raw, label_column=args.label_column, include_xgboost=args.include_xgboost)
    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if "error" not in report else 2


if __name__ == "__main__":
    raise SystemExit(main())
