import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importa las funciones de carga desde tu script existente
from load_data import load_full_dataset, load_activity_labels

# Directorio de salida
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "variables_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración general 
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})
sns.set_theme(style="whitegrid", palette="Set2")


# ── 1. Conteo de muestras por actividad ───────────────────────────────────────
def plot_class_balance(df):
    activity_order = df["activity_name"].value_counts().index.tolist()
    counts = df["activity_name"].value_counts().reindex(activity_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(counts.index, counts.values,
                   color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Número de muestras")
    ax.set_title("Distribución de clases", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/class_balance.png")


# ── 2. Distribución KDE por actividad ─────────────────────────────────────────
def plot_distributions(df, features, ncols=3):
    """KDE de cada feature desglosada por actividad."""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, features):
        for activity, grp in df.groupby("activity_name"):
            grp[feat].plot.kde(ax=ax, label=activity, linewidth=1.8)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=6)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Distribución de features por actividad",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dist_features.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/dist_features.png")


# ── 3. Boxplots comparativos por actividad ────────────────────────────────────
def plot_boxplots(df, features, ncols=3):
    """Boxplot de cada feature agrupado por actividad."""
    activity_order = sorted(df["activity_name"].unique())
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    palette = sns.color_palette("Set2", len(activity_order))

    for ax, feat in zip(axes, features):
        sns.boxplot(
            data=df,
            x="activity_name",
            y=feat,
            order=activity_order,
            palette=palette,
            width=0.55,
            fliersize=2,
            ax=ax,
        )
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=40, labelsize=7)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Comparativa por actividad (boxplot)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplots_features.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/boxplots_features.png")


# ── 4. Heatmap de correlación ─────────────────────────────────────────────────
def plot_correlation(df, features, method="pearson"):
    """Heatmap de correlación entre las features seleccionadas."""
    corr = df[features].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(features)), max(6, len(features) - 1)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=len(features) <= 20,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        square=True,
        ax=ax,
    )
    ax.set_title(f"Correlación ({method}) entre features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/correlation_heatmap.png")



# ── 5. Heatmap de medias por actividad ────────────────────────────────────────
def plot_means_heatmap(df, features=None):
    """
    Heatmap con la media normalizada de cada feature (columnas) por actividad (filas).
    Si no se pasa features, usa todas las columnas numéricas.
    """
    numeric_cols = df.select_dtypes(include="number").columns.difference(
        ["activity_id", "subject", "target"]
    )
    cols = features if features is not None else list(numeric_cols)

    means = df.groupby("activity_name")[cols].mean()

    # Normalizar por columna (z-score) para que sean comparables entre features
    means_norm = (means - means.mean()) / (means.std() + 1e-8)

    fig, ax = plt.subplots(figsize=(max(14, len(cols) // 4), 5))
    sns.heatmap(
        means_norm,
        cmap="RdBu_r",
        center=0,
        linewidths=0.2,
        ax=ax,
        cbar_kws={"label": "z-score respecto a la media global"},
    )
    ax.set_title("Media normalizada por actividad y feature",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "means_heatmap.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/means_heatmap.png")


# ── 6. Serie temporal de una feature por actividad ────────────────────────────
def plot_time_series(df, feature, n_samples=200):
    """
    Muestra cómo evoluciona una feature a lo largo de las muestras consecutivas,
    coloreando cada segmento según la actividad.
    n_samples: número de muestras consecutivas a mostrar (para no saturar el gráfico).
    """
    activity_order = sorted(df["activity_name"].unique())
    palette = dict(zip(activity_order, sns.color_palette("Set2", len(activity_order))))

    subset = df[[feature, "activity_name"]].head(n_samples).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Dibujar segmento a segmento para colorear por actividad
    prev_activity = subset.loc[0, "activity_name"]
    seg_start = 0
    plotted_labels = set()

    for i in range(1, len(subset)):
        curr_activity = subset.loc[i, "activity_name"]
        if curr_activity != prev_activity or i == len(subset) - 1:
            end = i if curr_activity != prev_activity else i + 1
            label = prev_activity if prev_activity not in plotted_labels else "_nolegend_"
            plotted_labels.add(prev_activity)
            ax.plot(
                range(seg_start, end),
                subset.loc[seg_start:end - 1, feature],
                color=palette[prev_activity],
                label=label,
                linewidth=1.5,
            )
            seg_start = i
            prev_activity = curr_activity

    ax.set_title(f"Serie temporal — {feature} (primeras {n_samples} muestras)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel(feature)
    ax.legend(title="Actividad", fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_series.png", bbox_inches="tight")
    plt.show()
    print("✔ Guardado: results/variables_comparison/time_series.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando datos...")
    train_df, test_df = load_full_dataset()
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → {len(df):,} muestras | {df.shape[1] - 3} features\n")

    # ── Selección de features a visualizar ──────────────────────────────────
    # Los nombres usan el formato de make_unique(): duplicados llevan __1, __2...

    # Opción A — lista manual (las más interpretables del dataset):
    FEATURES_TO_PLOT = [
        "tBodyAcc-mean()-X",
        "tBodyAcc-mean()-Y",
        "tBodyAcc-mean()-Z",
        "tBodyAcc-std()-X",
        "tBodyGyro-mean()-X",
        "tBodyGyro-mean()-Y",
        "tBodyGyro-mean()-Z",
        "tGravityAcc-mean()-X",
        "tBodyAccJerk-mean()-X",
    ]

    # Opción B — todas las features que contengan 'mean()' (descomenta):
    # FEATURES_TO_PLOT = [c for c in df.columns if "mean()" in c]

    # ── Gráficas ─────────────────────────────────────────────────────────────
    print("1/6 → Conteo de clases...")
    plot_class_balance(df)

    print("2/6 → Distribuciones KDE...")
    plot_distributions(df, FEATURES_TO_PLOT)

    print("3/6 → Boxplots...")
    plot_boxplots(df, FEATURES_TO_PLOT)

    print("4/6 → Correlación...")
    plot_correlation(df, FEATURES_TO_PLOT, method="pearson")

    print("5/6 → Heatmap de medias por actividad (todas las features)...")
    plot_means_heatmap(df)   # pasa features=FEATURES_TO_PLOT para limitar a la selección

    print("6/6 → Serie temporal...")
    plot_time_series(df, feature="tBodyAcc-mean()-X", n_samples=300)
    # Cambia 'feature' y 'n_samples' según lo que quieras explorar

    print("\n✅ ¡Listo! Imágenes guardadas en results/variables_comparison/")