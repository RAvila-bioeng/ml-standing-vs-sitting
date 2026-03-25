import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importa las funciones de carga desde tu script 'load_data.py'
from load_data import load_full_dataset

# ── CONFIGURACIÓN DE DIRECTORIOS ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "variables_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración estética global
plt.rcParams.update({"figure.dpi": 100, "figure.facecolor": "white"})
sns.set_theme(style="whitegrid", palette="Set2")


# ── 1. CONTEO DE CLASES ───────────────────────────────────────────────────────
def plot_class_balance(df):
    print("  → Generando: Distribución de clases...")
    counts = df["activity_name"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Número de muestras")
    ax.set_title("Distribución de Clases en el Dataset", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_class_balance.png", bbox_inches="tight")
    plt.close()


# ── 2. PCA: ANÁLISIS DE COMPONENTES PRINCIPALES (NUEVO) ───────────────────────
def plot_pca(df, n_components=2):
    print(f"  → Generando: PCA ({n_components}D)...")
    
    # Seleccionar solo columnas numéricas de sensores
    features_cols = df.select_dtypes(include="number").columns.difference(
        ["activity_id", "subject", "target"]
    )
    x = df[features_cols].values
    
    # El escalado es obligatorio para PCA
    x_scaled = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(x_scaled)
    
    pca_df = pd.DataFrame(
        data=components, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df["activity"] = df["activity_name"].values
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="activity",
        alpha=0.5, s=30, palette="Set2", edgecolor="none"
    )
    
    var_exp = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var_exp[0]:.2%} varianza)")
    plt.ylabel(f"PC2 ({var_exp[1]:.2%} varianza)")
    plt.title("PCA: Proyección del Dataset (Separabilidad de Clases)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Actividad")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_pca_analysis.png", bbox_inches="tight")
    plt.close()


# ── 3. DISTRIBUCIONES KDE ─────────────────────────────────────────────────────
def plot_distributions(df, features, ncols=3):
    print("  → Generando: Distribuciones KDE...")
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, features):
        # Optimizamos usando sns.kdeplot directamente
        sns.kdeplot(data=df, x=feat, hue="activity_name", ax=ax, fill=True, common_norm=False, alpha=0.3)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.get_legend().remove() if ax.get_legend() else None

    # Añadir una leyenda única para todo el gráfico
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=10)

    for ax in axes[len(features):]: ax.set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "3_dist_features.png", bbox_inches="tight")
    plt.close()


# ── 4. BOXPLOTS ───────────────────────────────────────────────────────────────
def plot_boxplots(df, features, ncols=3):
    print("  → Generando: Boxplots comparativos...")
    activity_order = sorted(df["activity_name"].unique())
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, features):
        sns.boxplot(data=df, x="activity_name", y=feat, order=activity_order, ax=ax, fliersize=1)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for ax in axes[len(features):]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_boxplots_features.png", bbox_inches="tight")
    plt.close()


# ── 5. CORRELACIÓN ────────────────────────────────────────────────────────────
def plot_correlation(df, features):
    print("  → Generando: Heatmap de correlación...")
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=len(features) <= 15, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Correlación entre Features Seleccionadas", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_correlation_heatmap.png", bbox_inches="tight")
    plt.close()


# ── 6. HEATMAP DE MEDIAS (Z-SCORE) ───────────────────────────────────────────
def plot_means_heatmap(df, features=None):
    print("  → Generando: Heatmap de medias normalizadas...")
    numeric_cols = df.select_dtypes(include="number").columns.difference(["activity_id", "subject", "target"])
    cols = features if features is not None else list(numeric_cols[:50]) # Limitamos si no hay selección

    means = df.groupby("activity_name")[cols].mean()
    means_norm = (means - means.mean()) / (means.std() + 1e-8)

    plt.figure(figsize=(15, 6))
    sns.heatmap(means_norm, cmap="RdBu_r", center=0, linewidths=0.1, cbar_kws={"label": "z-score"})
    plt.title("Perfil Medio de Actividad (Variables Normalizadas)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "6_means_heatmap.png", bbox_inches="tight")
    plt.close()


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Iniciando Análisis Visual...")
    
    # 1. Cargar datos
    train_df, test_df = load_full_dataset()
    df_full = pd.concat([train_df, test_df], ignore_index=True)
    
    # 2. Selección estratégica de variables para gráficas detalladas
    # He incluido variables de cuerpo, gravedad y frecuencia para tener variedad
    SELECTED_FEATURES = [
        "tBodyAcc-mean()-X", 
        "tGravityAcc-mean()-X", 
        "tBodyGyro-mean()-X",
        "tBodyAccJerk-std()-X",
        "fBodyAcc-mean()-X",
        "fBodyAccMag-mean()"
    ]

    # 3. Ejecutar funciones de graficado
    plot_class_balance(df_full)
    plot_pca(df_full) # Análisis de dimensionalidad
    plot_distributions(df_full, SELECTED_FEATURES)
    plot_boxplots(df_full, SELECTED_FEATURES)
    plot_correlation(df_full, SELECTED_FEATURES)
    plot_means_heatmap(df_full, features=SELECTED_FEATURES + [c for c in df_full.columns if "entropy" in c][:10])

    print(f"\n✅ Proceso finalizado. Gráficas guardadas en: {OUTPUT_DIR}")