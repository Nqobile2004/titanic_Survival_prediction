"""
Titanic Survival Prediction
----------------------------
End-to-end pipeline: data cleaning, feature engineering, model training,
cross-validation, and hyperparameter tuning using scikit-learn's
RandomForestClassifier.

Data source: download train.csv from the Kaggle Titanic competition
https://www.kaggle.com/c/titanic/data and place it in the same folder
as this script.

Usage:
    pip install pandas scikit-learn
    python titanic_survival_prediction.py
"""

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(path: str = "train.csv") -> pd.DataFrame:
    """Load the Titanic dataset and handle missing values."""
    df = pd.read_csv(path)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and encode categorical columns."""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])          # male=1, female=0
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    return df


def train_baseline(X_train, y_train) -> RandomForestClassifier:
    """Train a baseline Random Forest with default-ish hyperparameters."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def run_cross_validation(model, X, y, n_splits: int = 5) -> None:
    """Run stratified k-fold cross-validation and print the results."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print("\n--- Cross-Validation ---")
    print("CV scores:", scores)
    print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")


def tune_hyperparameters(X_train, y_train) -> GridSearchCV:
    """Search for the best Random Forest hyperparameters via GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning ---")
    print("Best params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    return grid_search


def evaluate(model, X_test, y_test, label: str = "Model") -> None:
    """Print accuracy, classification report, and confusion matrix."""
    preds = model.predict(X_test)

    print(f"\n--- {label} Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))


def main():
    df = load_and_clean_data("train.csv")
    df = engineer_features(df)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline model
    baseline_model = train_baseline(X_train, y_train)
    evaluate(baseline_model, X_test, y_test, label="Baseline")

    # Cross-validation on the baseline
    run_cross_validation(baseline_model, X, y)

    # Hyperparameter tuning
    grid_search = tune_hyperparameters(X_train, y_train)
    best_model = grid_search.best_estimator_
    evaluate(best_model, X_test, y_test, label="Tuned")

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\n--- Feature Importance ---")
    print(importances.sort_values(ascending=False))


if __name__ == "__main__":
    main()
