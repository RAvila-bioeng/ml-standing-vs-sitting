from pathlib import Path
import pandas as pd


DATASET_DIR = Path("data/raw/UCI HAR Dataset/UCI HAR Dataset")


def make_unique(names: list[str]) -> list[str]:
    counts = {}
    unique_names = []

    for name in names:
        if name not in counts:
            counts[name] = 0
            unique_names.append(name)
        else:
            counts[name] += 1
            unique_names.append(f"{name}__{counts[name]}")

    return unique_names


def load_features():
    features_path = DATASET_DIR / "features.txt"
    features = pd.read_csv(
        features_path,
        sep=r"\s+",
        header=None,
        names=["index", "feature"]
    )
    feature_names = features["feature"].tolist()
    return make_unique(feature_names)


def load_activity_labels():
    labels_path = DATASET_DIR / "activity_labels.txt"
    activity_labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity_name"]
    )
    return dict(zip(activity_labels["activity_id"], activity_labels["activity_name"]))


def load_split(split: str) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    feature_names = load_features()

    x_path = DATASET_DIR / split / f"X_{split}.txt"
    y_path = DATASET_DIR / split / f"y_{split}.txt"
    subject_path = DATASET_DIR / split / f"subject_{split}.txt"

    X = pd.read_csv(x_path, sep=r"\s+", header=None)
    X.columns = feature_names

    y = pd.read_csv(y_path, sep=r"\s+", header=None, names=["activity_id"])
    subject = pd.read_csv(subject_path, sep=r"\s+", header=None, names=["subject"])

    df = pd.concat([subject, y, X], axis=1)

    activity_map = load_activity_labels()
    df["activity_name"] = df["activity_id"].map(activity_map)

    return df


def load_full_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_split("train")
    test_df = load_split("test")
    return train_df, test_df


def filter_sitting_standing(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["activity_name"].isin(["SITTING", "STANDING"])].copy()
    filtered["target"] = filtered["activity_name"].map({"SITTING": 0, "STANDING": 1})
    return filtered


if __name__ == "__main__":
    train_df, test_df = load_full_dataset()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain activity counts:")
    print(train_df["activity_name"].value_counts())

    filtered_train = filter_sitting_standing(train_df)
    filtered_test = filter_sitting_standing(test_df)

    print("\nFiltered train shape:", filtered_train.shape)
    print("Filtered test shape:", filtered_test.shape)

    print("\nFiltered train counts:")
    print(filtered_train["activity_name"].value_counts())

    print("\nFiltered test counts:")
    print(filtered_test["activity_name"].value_counts())