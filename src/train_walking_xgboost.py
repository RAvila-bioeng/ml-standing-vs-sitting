import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from load_data import PROJECT_ROOT, filter_activity_subset, load_full_dataset

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost no esta instalado en el entorno. Instala la dependencia para ejecutar este script."
    ) from exc


OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
TARGET_MAP = {activity: idx for idx, activity in enumerate(TARGET_ACTIVITIES)}


def prepare_data():
    train_df, test_df = load_full_dataset()

    train_df = filter_activity_subset(train_df, TARGET_ACTIVITIES)
    test_df = filter_activity_subset(test_df, TARGET_ACTIVITIES)

    feature_cols = [
        col for col in train_df.columns if col not in ["subject", "activity_id", "activity_name"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["activity_name"].map(TARGET_MAP)
    X_test = test_df[feature_cols]
    y_test = test_df["activity_name"].map(TARGET_MAP)

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TARGET_ACTIVITIES,
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusion con XGBoost para actividades walking")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "walking_xgboost_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    return cm


def build_model():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(TARGET_ACTIVITIES),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = plot_confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=TARGET_ACTIVITIES,
    )

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    with open(OUTPUT_DIR / "walking_xgboost_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification report:\n")
        f.write(report)


if __name__ == "__main__":
    main()
