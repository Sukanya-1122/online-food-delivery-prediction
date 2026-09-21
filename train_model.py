"""Train and compare binary classifiers for online food delivery prediction."""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "Output"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Marital Status",
    "Occupation",
    "Monthly Income",
    "Educational Qualifications",
    "Family size",
    "Customer Type",
    "latitude",
    "longitude",
    "Pin code",
    "Feedback",
]
NUMERIC_COLUMNS = ["Age", "Family size", "latitude", "longitude", "Pin code"]
CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV, remove index-like columns, normalize text, and deduplicate."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[column for column in df.columns if column.startswith("Unnamed:")], errors="ignore")

    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].drop_duplicates().copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.title()
    df = df[df[TARGET_COLUMN].isin(["Yes", "No"])].copy()
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, labels=["No", "Yes"], zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
    print(f"\n{name}")
    print("Confusion Matrix (rows=true, columns=predicted; labels=No, Yes):")
    print(matrix)
    print("Classification Report:")
    print(report)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label="Yes", zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label="Yes", zero_division=0),
        "F1-Score": f1_score(y_test, predictions, pos_label="Yes", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the online food delivery classifier.")
    parser.add_argument("--data", default="data/online food delivery dataset.csv", help="Path to the input CSV")
    parser.add_argument("--model-output", default="models/food_delivery_model.pkl", help="Path for the joblib artifact")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model_output)
    df = load_and_clean_data(data_path)

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Duplicate rows after cleaning: {df.duplicated().sum()}")
    print(f"Target distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    print(f"Missing values:\n{df.isna().sum().to_string()}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    evaluation_rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        evaluation_rows.append(evaluate_model(name, pipeline, X_test, y_test))

    comparison = pd.DataFrame(evaluation_rows).sort_values(
        by=["F1-Score", "Recall", "Precision", "Accuracy"], ascending=False
    )
    print("\nModel comparison (positive class: Yes):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    selected_name = comparison.iloc[0]["Model"]
    selected_model = trained_models[selected_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categorical_options": {
            column: sorted(df[column].dropna().unique().tolist()) for column in CATEGORICAL_COLUMNS
        },
        "numeric_defaults": {
            column: float(df[column].median()) for column in NUMERIC_COLUMNS
        },
        "numeric_ranges": {
            column: {"min": float(df[column].min()), "max": float(df[column].max())}
            for column in NUMERIC_COLUMNS
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    joblib.dump(artifact, model_path)
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model artifact to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
