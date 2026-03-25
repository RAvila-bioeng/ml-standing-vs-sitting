import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, filter_activity_pair, load_full_dataset


OUTPUT_DIR = PROJECT_ROOT / "results" / "sitting_vs_laying_insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["SITTING", "LAYING"]
CLASS_COLORS = {"SITTING": "#E45756", "LAYING": "#72B7B2"}
META_COLUMNS = ["subject", "activity_id", "activity_name", "target"]


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
    return counts.reindex(CLASS_ORDER)


def pick_top_features(df: pd.DataFrame, feature_cols: list[str], top_n: int = 6) -> list[str]:
    mean_by_activity = df.groupby("activity_name", observed=True)[feature_cols].mean()
    deltas = (mean_by_activity.loc["LAYING"] - mean_by_activity.loc["SITTING"]).abs().sort_values(ascending=False)
    return deltas.head(top_n).index.tolist()


def pick_subject_for_plot(df: pd.DataFrame) -> int:
    subject_counts = df.groupby(["subject", "activity_name"], observed=True).size().unstack(fill_value=0)
    eligible_subjects = subject_counts[(subject_counts["SITTING"] >= 40) & (subject_counts["LAYING"] >= 40)]
    if not eligible_subjects.empty:
        return int(eligible_subjects.sum(axis=1).idxmax())
    return int(subject_counts.sum(axis=1).idxmax())


def plot_activity_distribution(counts: pd.DataFrame):
    ax = counts.plot(kind="bar", figsize=(7, 4), color=["#4C78A8", "#F58518"])
    ax.set_title("Distribucion de muestras para SITTING y LAYING")
    ax.set_xlabel("Actividad")
    ax.set_ylabel("Numero de muestras")
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)

    save_current_figure("activity_distribution.png")


def plot_pca_projection(df: pd.DataFrame, feature_cols: list[str]):
    scaled = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(8, 6))
    for activity in CLASS_ORDER:
        activity_mask = df["activity_name"] == activity
        plt.scatter(
            components[activity_mask, 0],
            components[activity_mask, 1],
            s=16,
            alpha=0.55,
            label=activity,
            c=CLASS_COLORS[activity],
        )

    plt.title("PCA para SITTING vs LAYING")
    plt.xlabel(f"PC1 ({explained_variance[0]:.1f}% de varianza explicada)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1f}% de varianza explicada)")
    plt.legend(loc="best", fontsize=8)

    save_current_figure("pca_projection.png")


def plot_boxplots(df: pd.DataFrame, selected_features: list[str]):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        data = [df.loc[df["activity_name"] == activity, feature] for activity in CLASS_ORDER]
        box = ax.boxplot(data, patch_artist=True, tick_labels=CLASS_ORDER)
        for patch, activity in zip(box["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[activity])
            patch.set_alpha(0.7)
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(len(selected_features), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Features que mejor separan SITTING y LAYING", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_feature_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_subject_signals(df: pd.DataFrame, selected_features: list[str]):
    subject_id = pick_subject_for_plot(df)
    subject_df = df[df["subject"] == subject_id].copy()
    features_to_plot = selected_features[: min(3, len(selected_features))]

    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(12, 8), sharex=False)
    if len(features_to_plot) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features_to_plot):
        start = 0
        tick_positions = []
        tick_labels = []

        for activity in CLASS_ORDER:
            values = subject_df.loc[subject_df["activity_name"] == activity, feature].reset_index(drop=True)
            x_values = range(start, start + len(values))
            ax.plot(
                list(x_values),
                values,
                color=CLASS_COLORS[activity],
                linewidth=2,
                label=activity,
            )
            if len(values) > 0:
                ax.axvspan(start, start + len(values) - 1, color=CLASS_COLORS[activity], alpha=0.08)
                tick_positions.append(start + (len(values) // 2))
                tick_labels.append(activity)
            start += len(values)

        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("Valor")
        ax.set_xticks(tick_positions, tick_labels)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel(f"Muestras consecutivas del sujeto {subject_id}")
    fig.suptitle(f"Comparacion secuencial entre SITTING y LAYING para el sujeto {subject_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subject_signals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, selected_features: list[str]):
    counts = get_activity_counts(train_df, test_df)
    subject_id = pick_subject_for_plot(pd.concat([train_df, test_df], ignore_index=True))

    with open(OUTPUT_DIR / "dataset_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Sitting vs laying dataset summary\n")
        handle.write("===============================\n\n")
        handle.write(f"Train shape: {train_df.shape}\n")
        handle.write(f"Test shape: {test_df.shape}\n")
        handle.write(f"Combined shape: {pd.concat([train_df, test_df], ignore_index=True).shape}\n")
        handle.write("\nSamples by activity\n")
        handle.write("-------------------\n")
        handle.write(counts.fillna(0).astype(int).to_string())
        handle.write("\n\nBest features for SITTING vs LAYING\n")
        handle.write("-----------------------------------\n")
        for feature in selected_features:
            handle.write(f"- {feature}\n")
        handle.write(f"\nSubject used for sequential plot: {subject_id}\n")


def main():
    train_df, test_df = load_full_dataset()
    train_df = filter_activity_pair(train_df, "SITTING", "LAYING")
    test_df = filter_activity_pair(test_df, "SITTING", "LAYING")
    df = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = get_feature_columns(df)
    selected_features = pick_top_features(df, feature_cols)

    plot_activity_distribution(get_activity_counts(train_df, test_df))
    plot_pca_projection(df, feature_cols)
    plot_boxplots(df, selected_features)
    plot_subject_signals(df, selected_features)
    write_summary(train_df, test_df, selected_features)

    print(f"Insights guardados en: {OUTPUT_DIR}")
    print("Plots generados:")
    for path in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
