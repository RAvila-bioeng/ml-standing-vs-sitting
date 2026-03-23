from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, filter_sitting_standing, load_full_dataset


OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_data():
    train_df, test_df = load_full_dataset()

    train_df = filter_sitting_standing(train_df)
    test_df = filter_sitting_standing(test_df)

    feature_cols = [
        col for col in train_df.columns
        if col not in ["subject", "activity_id", "activity_name", "target"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["SITTING", "STANDING"],
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusion del baseline")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    return cm


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = plot_confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["SITTING", "STANDING"]
    )

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    with open(OUTPUT_DIR / "baseline_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification report:\n")
        f.write(report)


if __name__ == "__main__":
    main()
