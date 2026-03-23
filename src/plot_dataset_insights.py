from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, filter_sitting_standing, load_full_dataset


OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLUMNS = ["subject", "activity_id", "activity_name"]
ACTIVITY_ORDER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
ACTIVITY_COLORS = {
    "WALKING": "#4C78A8",
    "WALKING_UPSTAIRS": "#72B7B2",
    "WALKING_DOWNSTAIRS": "#54A24B",
    "SITTING": "#E45756",
    "STANDING": "#F58518",
    "LAYING": "#B279A2",
}


def save_current_figure(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS]


def get_activity_counts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    counts = pd.DataFrame(
        {
            "train": train_df["activity_name"].value_counts(),
            "test": test_df["activity_name"].value_counts(),
        }
    )
    return counts.reindex(ACTIVITY_ORDER)


def summarize_feature_groups(feature_cols: list[str]) -> pd.DataFrame:
    summary = pd.DataFrame({"feature": feature_cols})
    summary["domain"] = summary["feature"].str.extract(r"^(t|f)")
    summary["sensor"] = summary["feature"].str.extract(r"(BodyAcc|GravityAcc|BodyGyro)")

    domain_counts = summary["domain"].map({"t": "time_domain", "f": "frequency_domain"}).value_counts()
    sensor_counts = summary["sensor"].fillna("other").value_counts()

    max_len = max(len(domain_counts), len(sensor_counts))
    domain_values = list(domain_counts.items()) + [("", "")] * (max_len - len(domain_counts))
    sensor_values = list(sensor_counts.items()) + [("", "")] * (max_len - len(sensor_counts))

    return pd.DataFrame(
        {
            "domain_group": [item[0] for item in domain_values],
            "domain_count": [item[1] for item in domain_values],
            "sensor_group": [item[0] for item in sensor_values],
            "sensor_count": [item[1] for item in sensor_values],
        }
    )


def pick_representative_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    preferred = [
        "tBodyAcc-mean()-X",
        "tBodyAcc-mean()-Y",
        "tBodyAcc-mean()-Z",
        "tGravityAcc-mean()-X",
        "tGravityAcc-mean()-Y",
        "tGravityAcc-mean()-Z",
        "tBodyGyro-mean()-X",
        "tBodyGyro-mean()-Y",
        "tBodyGyro-mean()-Z",
    ]
    selected = [feature for feature in preferred if feature in feature_cols]

    if len(selected) >= 6:
        return selected[:6]

    variances = df[feature_cols].var().sort_values(ascending=False)
    for feature in variances.index:
        if feature not in selected:
            selected.append(feature)
        if len(selected) == 6:
            break

    return selected


def pick_sitting_standing_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 4) -> list[str]:
    sitting_standing_df = filter_sitting_standing(df)
    mean_by_activity = sitting_standing_df.groupby("activity_name")[feature_cols].mean().T
    return (
        (mean_by_activity["STANDING"] - mean_by_activity["SITTING"])
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )


def plot_activity_distribution(counts: pd.DataFrame):
    ax = counts.plot(
        kind="bar",
        figsize=(10, 5),
        color=["#4C78A8", "#F58518"],
    )
    ax.set_title("Distribucion de muestras por actividad")
    ax.set_xlabel("Actividad")
    ax.set_ylabel("Numero de muestras")
    ax.tick_params(axis="x", rotation=20)

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)

    save_current_figure("activity_distribution.png")


def plot_pca_projection(df: pd.DataFrame, feature_cols: list[str]):
    samples = []
    for activity, group in df.groupby("activity_name"):
        samples.append(group.sample(n=min(len(group), 400), random_state=42))

    sample_df = pd.concat(samples, ignore_index=True)

    scaled = StandardScaler().fit_transform(sample_df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(9, 7))
    for activity in ACTIVITY_ORDER:
        activity_mask = sample_df["activity_name"] == activity
        plt.scatter(
            components[activity_mask, 0],
            components[activity_mask, 1],
            s=18,
            alpha=0.65,
            label=activity,
            c=ACTIVITY_COLORS[activity],
        )

    plt.title("Proyeccion PCA del dataset completo")
    plt.xlabel(f"PC1 ({explained_variance[0]:.1f}% de varianza explicada)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1f}% de varianza explicada)")
    plt.legend(loc="best", fontsize=8)

    save_current_figure("pca_projection.png")


def plot_top_variable_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 15):
    variances = df[feature_cols].var().sort_values(ascending=False).head(top_n).sort_values()

    ax = variances.plot(kind="barh", figsize=(11, 7), color="#54A24B")
    ax.set_title("Features con mayor variabilidad global")
    ax.set_xlabel("Varianza")
    ax.set_ylabel("Feature")

    save_current_figure("top_variable_features.png")


def plot_feature_distributions(df: pd.DataFrame, selected_features: list[str]):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        activity_means = df.groupby("activity_name")[feature].mean().reindex(ACTIVITY_ORDER)
        ax.bar(
            activity_means.index,
            activity_means.values,
            color=[ACTIVITY_COLORS[activity] for activity in activity_means.index],
        )
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(len(selected_features), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Media por actividad en variables representativas", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_feature_means.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation(df: pd.DataFrame, selected_features: list[str]):
    corr = df[selected_features].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlacion")
    plt.title("Correlacion entre variables representativas")
    plt.xticks(range(len(selected_features)), selected_features, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(selected_features)), selected_features, fontsize=8)

    save_current_figure("representative_feature_correlation.png")


def plot_sitting_standing_feature_boxplots(df: pd.DataFrame, selected_features: list[str]):
    filtered_df = filter_sitting_standing(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    class_order = ["SITTING", "STANDING"]
    class_colors = {"SITTING": "#E45756", "STANDING": "#F58518"}

    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        data = [
            filtered_df.loc[filtered_df["activity_name"] == activity, feature]
            for activity in class_order
        ]
        box = ax.boxplot(data, patch_artist=True, tick_labels=class_order)
        for patch, activity in zip(box["boxes"], class_order):
            patch.set_facecolor(class_colors[activity])
            patch.set_alpha(0.7)

        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(len(selected_features), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Variables que mejor separan SITTING y STANDING", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sitting_vs_standing_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pick_subject_for_sitting_standing_plot(df: pd.DataFrame) -> int:
    filtered_df = filter_sitting_standing(df)
    subject_counts = (
        filtered_df.groupby(["subject", "activity_name"])
        .size()
        .unstack(fill_value=0)
    )
    eligible_subjects = subject_counts[
        (subject_counts["SITTING"] >= 60) & (subject_counts["STANDING"] >= 60)
    ]

    if not eligible_subjects.empty:
        return int(eligible_subjects.sum(axis=1).idxmax())

    return int(subject_counts.sum(axis=1).idxmax())


def plot_subject_sitting_standing_signals(df: pd.DataFrame, selected_features: list[str]):
    filtered_df = filter_sitting_standing(df)
    subject_id = pick_subject_for_sitting_standing_plot(df)
    subject_df = filtered_df[filtered_df["subject"] == subject_id].copy()
    class_order = ["SITTING", "STANDING"]
    features_to_plot = selected_features[: min(3, len(selected_features))]

    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(12, 8), sharex=False)
    if len(features_to_plot) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features_to_plot):
        start = 0
        tick_positions = []
        tick_labels = []

        for activity in class_order:
            values = (
                subject_df.loc[subject_df["activity_name"] == activity, feature]
                .reset_index(drop=True)
            )
            x_values = range(start, start + len(values))
            ax.plot(
                list(x_values),
                values,
                color=ACTIVITY_COLORS[activity],
                linewidth=2,
                label=activity,
            )
            ax.axvspan(start, start + len(values) - 1, color=ACTIVITY_COLORS[activity], alpha=0.08)
            tick_positions.append(start + (len(values) // 2))
            tick_labels.append(activity)
            start += len(values)

        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("Valor")
        ax.set_xticks(tick_positions, tick_labels)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel(f"Muestras consecutivas del sujeto {subject_id}")
    fig.suptitle(
        f"Comparacion secuencial entre SITTING y STANDING para el sujeto {subject_id}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subject_sitting_vs_standing_signals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    counts = get_activity_counts(train_df, test_df)
    feature_groups = summarize_feature_groups(feature_cols)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    sitting_standing_features = pick_sitting_standing_features(combined_df, feature_cols)
    subject_id = pick_subject_for_sitting_standing_plot(combined_df)

    with open(OUTPUT_DIR / "dataset_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Dataset summary\n")
        handle.write("================\n\n")
        handle.write(f"Train shape: {train_df.shape}\n")
        handle.write(f"Test shape: {test_df.shape}\n")
        handle.write(f"Combined shape: {(pd.concat([train_df, test_df]).shape)}\n")
        handle.write(f"Number of subjects: {pd.concat([train_df, test_df])['subject'].nunique()}\n")
        handle.write(f"Number of features: {len(feature_cols)}\n\n")
        handle.write("Samples by activity\n")
        handle.write("-------------------\n")
        handle.write(counts.fillna(0).astype(int).to_string())
        handle.write("\n\nFeature groups\n")
        handle.write("--------------\n")
        handle.write(feature_groups.to_string(index=False))
        handle.write("\n\nBest features for SITTING vs STANDING\n")
        handle.write("-------------------------------------\n")
        for feature in sitting_standing_features:
            handle.write(f"- {feature}\n")
        handle.write(f"\nSubject used for sequential plot: {subject_id}\n")


def main():
    train_df, test_df = load_full_dataset()
    df = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = get_feature_columns(df)
    selected_features = pick_representative_features(df, feature_cols)
    sitting_standing_features = pick_sitting_standing_features(df, feature_cols)

    plot_activity_distribution(get_activity_counts(train_df, test_df))
    plot_pca_projection(df, feature_cols)
    plot_top_variable_features(df, feature_cols)
    plot_feature_distributions(df, selected_features)
    plot_feature_correlation(df, selected_features)
    plot_sitting_standing_feature_boxplots(df, sitting_standing_features)
    plot_subject_sitting_standing_signals(df, sitting_standing_features)
    write_summary(train_df, test_df, feature_cols)

    print(f"Dataset insights guardados en: {OUTPUT_DIR}")
    print("Plots generados:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
