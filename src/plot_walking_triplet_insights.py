import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, filter_activity_subset, load_full_dataset


OUTPUT_DIR = PROJECT_ROOT / "results" / "walking_triplet_insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
ACTIVITY_COLORS = {
    "WALKING": "#4C78A8",
    "WALKING_UPSTAIRS": "#72B7B2",
    "WALKING_DOWNSTAIRS": "#54A24B",
}
META_COLUMNS = ["subject", "activity_id", "activity_name"]


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
    return counts.reindex(TARGET_ACTIVITIES)


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


def pick_top_triplet_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 6) -> list[str]:
    mean_by_activity = df.groupby("activity_name", observed=True)[feature_cols].mean()
    spread = (mean_by_activity.max(axis=0) - mean_by_activity.min(axis=0)).sort_values(ascending=False)
    return spread.head(top_n).index.tolist()


def pick_walking_vs_stairs_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 6) -> list[str]:
    mean_by_activity = df.groupby("activity_name", observed=True)[feature_cols].mean()
    stairs_mean = (
        mean_by_activity.loc["WALKING_UPSTAIRS"] + mean_by_activity.loc["WALKING_DOWNSTAIRS"]
    ) / 2
    deltas = (mean_by_activity.loc["WALKING"] - stairs_mean).abs().sort_values(ascending=False)
    return deltas.head(top_n).index.tolist()


def pick_upstairs_vs_downstairs_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 6) -> list[str]:
    mean_by_activity = df.groupby("activity_name", observed=True)[feature_cols].mean()
    deltas = (
        mean_by_activity.loc["WALKING_UPSTAIRS"] - mean_by_activity.loc["WALKING_DOWNSTAIRS"]
    ).abs().sort_values(ascending=False)
    return deltas.head(top_n).index.tolist()


def pick_subject_for_triplet_plot(df: pd.DataFrame) -> int:
    subject_counts = df.groupby(["subject", "activity_name"], observed=True).size().unstack(fill_value=0)
    eligible_subjects = subject_counts[
        (subject_counts["WALKING"] >= 40)
        & (subject_counts["WALKING_UPSTAIRS"] >= 40)
        & (subject_counts["WALKING_DOWNSTAIRS"] >= 40)
    ]

    if not eligible_subjects.empty:
        return int(eligible_subjects.sum(axis=1).idxmax())

    return int(subject_counts.sum(axis=1).idxmax())


def plot_activity_distribution(counts: pd.DataFrame):
    ax = counts.plot(kind="bar", figsize=(10, 5), color=["#4C78A8", "#F58518"])
    ax.set_title("Distribucion de muestras para actividades de walking")
    ax.set_xlabel("Actividad")
    ax.set_ylabel("Numero de muestras")
    ax.tick_params(axis="x", rotation=15)

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)

    save_current_figure("walking_triplet_activity_distribution.png")


def plot_pca_projection(df: pd.DataFrame, feature_cols: list[str]):
    scaled = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(9, 7))
    for activity in TARGET_ACTIVITIES:
        activity_mask = df["activity_name"] == activity
        plt.scatter(
            components[activity_mask, 0],
            components[activity_mask, 1],
            s=16,
            alpha=0.55,
            label=activity,
            c=ACTIVITY_COLORS[activity],
        )

    plt.title("PCA para WALKING, WALKING_UPSTAIRS y WALKING_DOWNSTAIRS")
    plt.xlabel(f"PC1 ({explained_variance[0]:.1f}% de varianza explicada)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1f}% de varianza explicada)")
    plt.legend(loc="best", fontsize=8)

    save_current_figure("walking_triplet_pca_projection.png")


def plot_boxplots(df: pd.DataFrame, selected_features: list[str], filename: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        data = [df.loc[df["activity_name"] == activity, feature] for activity in TARGET_ACTIVITIES]
        box = ax.boxplot(data, patch_artist=True, tick_labels=TARGET_ACTIVITIES)
        for patch, activity in zip(box["boxes"], TARGET_ACTIVITIES):
            patch.set_facecolor(ACTIVITY_COLORS[activity])
            patch.set_alpha(0.7)

        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(len(selected_features), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_subject_triplet_signals(df: pd.DataFrame, selected_features: list[str]):
    subject_id = pick_subject_for_triplet_plot(df)
    subject_df = df[df["subject"] == subject_id].copy()
    features_to_plot = selected_features[: min(3, len(selected_features))]

    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(12, 9), sharex=False)
    if len(features_to_plot) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features_to_plot):
        start = 0
        tick_positions = []
        tick_labels = []

        for activity in TARGET_ACTIVITIES:
            values = subject_df.loc[subject_df["activity_name"] == activity, feature].reset_index(drop=True)
            x_values = range(start, start + len(values))
            ax.plot(
                list(x_values),
                values,
                color=ACTIVITY_COLORS[activity],
                linewidth=2,
                label=activity,
            )
            if len(values) > 0:
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
        f"Comparacion secuencial para actividades walking del sujeto {subject_id}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "walking_triplet_subject_signals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    triplet_features: list[str],
    walking_vs_stairs_features: list[str],
    upstairs_vs_downstairs_features: list[str],
):
    counts = get_activity_counts(train_df, test_df)
    feature_groups = summarize_feature_groups(feature_cols)
    subject_id = pick_subject_for_triplet_plot(pd.concat([train_df, test_df], ignore_index=True))

    with open(OUTPUT_DIR / "dataset_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Walking triplet dataset summary\n")
        handle.write("===============================\n\n")
        handle.write(f"Target activities: {', '.join(TARGET_ACTIVITIES)}\n")
        handle.write(f"Train shape: {train_df.shape}\n")
        handle.write(f"Test shape: {test_df.shape}\n")
        handle.write(f"Combined shape: {pd.concat([train_df, test_df], ignore_index=True).shape}\n")
        handle.write(f"Number of subjects: {pd.concat([train_df, test_df], ignore_index=True)['subject'].nunique()}\n")
        handle.write(f"Number of features: {len(feature_cols)}\n\n")
        handle.write("Samples by activity\n")
        handle.write("-------------------\n")
        handle.write(counts.fillna(0).astype(int).to_string())
        handle.write("\n\nFeature groups\n")
        handle.write("--------------\n")
        handle.write(feature_groups.to_string(index=False))
        handle.write("\n\nTop features for the three-class comparison\n")
        handle.write("-------------------------------------------\n")
        for feature in triplet_features:
            handle.write(f"- {feature}\n")
        handle.write("\nBest features for WALKING vs stairs block\n")
        handle.write("-----------------------------------------\n")
        for feature in walking_vs_stairs_features:
            handle.write(f"- {feature}\n")
        handle.write("\nBest features for WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS\n")
        handle.write("-------------------------------------------------------\n")
        for feature in upstairs_vs_downstairs_features:
            handle.write(f"- {feature}\n")
        handle.write(f"\nSubject used for sequential plot: {subject_id}\n")


def main():
    train_df, test_df = load_full_dataset()
    train_df = filter_activity_subset(train_df, TARGET_ACTIVITIES)
    test_df = filter_activity_subset(test_df, TARGET_ACTIVITIES)
    df = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = get_feature_columns(df)

    triplet_features = pick_top_triplet_features(df, feature_cols)
    walking_vs_stairs_features = pick_walking_vs_stairs_features(df, feature_cols)
    upstairs_vs_downstairs_features = pick_upstairs_vs_downstairs_features(df, feature_cols)

    plot_activity_distribution(get_activity_counts(train_df, test_df))
    plot_pca_projection(df, feature_cols)
    plot_boxplots(
        df,
        triplet_features,
        "walking_triplet_top_feature_boxplots.png",
        "Distribucion de las features mas discriminativas del triplete walking",
    )
    plot_boxplots(
        df,
        walking_vs_stairs_features,
        "walking_vs_stairs_boxplots.png",
        "Features que mejor separan WALKING del bloque de escaleras",
    )
    plot_boxplots(
        df,
        upstairs_vs_downstairs_features,
        "upstairs_vs_downstairs_with_walking_context_boxplots.png",
        "Features que mejor distinguen WALKING_UPSTAIRS y WALKING_DOWNSTAIRS con WALKING como referencia",
    )
    plot_subject_triplet_signals(df, triplet_features)
    write_summary(
        train_df,
        test_df,
        feature_cols,
        triplet_features,
        walking_vs_stairs_features,
        upstairs_vs_downstairs_features,
    )

    print(f"Dataset insights guardados en: {OUTPUT_DIR}")
    print("Plots generados:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
