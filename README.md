# Human Activity Recognition with Smartphone Sensors

Proyecto de análisis y clasificación de actividades humanas usando el dataset **UCI HAR Dataset**. El repositorio empezó centrado en `SITTING` vs `STANDING`, pero ahora incluye análisis exploratorios y modelos para varios subconjuntos del problema, además de un clasificador multiclase sobre las 6 actividades originales.

## Objetivo

Trabajar con señales derivadas de acelerómetro y giroscopio para:

- explorar qué variables separan mejor distintas actividades
- comparar subconjuntos concretos de clases
- entrenar modelos base y modelos más potentes
- guardar métricas y matrices de confusión en una estructura simple y reproducible

## Dataset

Fuente:
- **UCI Human Activity Recognition Using Smartphones**

Actividades disponibles:
- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

Ubicación esperada en este proyecto:
- `data/raw/UCI HAR Dataset/UCI HAR Dataset/`

## Estructura

```text
ml-standing-vs-sitting/
├── data/
│   └── raw/
├── notebooks/
├── results/
│   ├── dataset_insights/
│   ├── sitting_vs_laying_insights/
│   ├── walking_triplet_insights/
│   └── variables_comparison/
├── src/
│   ├── load_data.py
│   ├── plot_dataset_insights.py
│   ├── plot_sitting_vs_laying_insights.py
│   ├── plot_walking_triplet_insights.py
│   ├── comparacion_variables.py
│   ├── train_baseline.py
│   ├── train_triplet_classifier.py
│   ├── train_walking_xgboost.py
│   └── train_full_xgboost.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Instalación

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Nota:
- Los scripts con XGBoost requieren `xgboost`. Si no está disponible en tu entorno, instálalo manualmente con `pip install xgboost`.

## Carga de datos

El archivo `src/load_data.py` centraliza la carga del dataset y expone funciones para:

- cargar `train` y `test`
- mapear `activity_id` a `activity_name`
- filtrar subconjuntos de actividades
- construir datasets binarios o multicategoría

## Análisis exploratorio

### 1. SITTING vs STANDING vs WALKING

Script:
- `src/plot_dataset_insights.py`

Genera:
- distribución de muestras
- PCA del triplete
- boxplots con las 6 variables más discriminativas
- comparación `WALKING` vs bloque estático
- comparación `SITTING` vs `STANDING` con `WALKING` como referencia
- señales secuenciales de un sujeto
- resumen textual

Salida:
- `results/dataset_insights/`

Ejecución:

```powershell
.\.venv\Scripts\python.exe src\plot_dataset_insights.py
```

### 2. SITTING vs LAYING

Script:
- `src/plot_sitting_vs_laying_insights.py`

Salida:
- `results/sitting_vs_laying_insights/`

Ejecución:

```powershell
.\.venv\Scripts\python.exe src\plot_sitting_vs_laying_insights.py
```

### 3. WALKING vs WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS

Script:
- `src/plot_walking_triplet_insights.py`

Genera el mismo tipo de análisis que el triplete anterior, adaptado a actividades de marcha:
- distribución de muestras
- PCA
- variables más discriminativas del triplete walking
- `WALKING` vs bloque de escaleras
- `WALKING_UPSTAIRS` vs `WALKING_DOWNSTAIRS` con `WALKING` como referencia
- señales por sujeto
- resumen textual

Salida:
- `results/walking_triplet_insights/`

Ejecución:

```powershell
.\.venv\Scripts\python.exe src\plot_walking_triplet_insights.py
```

### 4. Comparación general de variables

Script:
- `src/comparacion_variables.py`

Salida:
- `results/variables_comparison/`

## Modelos

### 1. Baseline binario: SITTING vs STANDING

Script:
- `src/train_baseline.py`

Modelo:
- `StandardScaler + LogisticRegression`

Salida:
- `results/confusion_matrix.png`
- `results/baseline_metrics.txt`

### 2. Clasificador triplete: SITTING, STANDING, WALKING

Script:
- `src/train_triplet_classifier.py`

Modelo:
- `StandardScaler + LogisticRegression`

Salida:
- `results/triplet_confusion_matrix.png`
- `results/triplet_metrics.txt`

### 3. XGBoost para actividades de walking

Script:
- `src/train_walking_xgboost.py`

Configuración actual:
- clasifica `WALKING`, `WALKING_UPSTAIRS` y `WALKING_DOWNSTAIRS`
- usa una selección de 6 variables del análisis exploratorio walking

Salida:
- `results/walking_xgboost_confusion_matrix.png`
- `results/walking_xgboost_metrics.txt`

### 4. XGBoost completo sobre las 6 actividades

Script:
- `src/train_full_xgboost.py`

Clasifica:
- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

Salida:
- `results/full_xgboost_confusion_matrix.png`
- `results/full_xgboost_metrics.txt`

## Resultados actuales

Resumen de los modelos ya ejecutados:

- Baseline binario `SITTING` vs `STANDING`: problema difícil pero bien planteado para un primer modelo.
- Clasificador triplete `SITTING` / `STANDING` / `WALKING`: `accuracy` aproximada de `0.9612`.
- XGBoost con 6 actividades: `accuracy` aproximada de `0.9352`.

Patrones observados:

- `LAYING` se separa muy bien.
- `WALKING` y sus variantes tienen buena separabilidad, aunque `WALKING_UPSTAIRS` y `WALKING_DOWNSTAIRS` siguen siendo clases cercanas.
- La confusión más persistente aparece entre `SITTING` y `STANDING`.

## Comandos útiles

Ejecutar análisis de estáticas vs walking:

```powershell
.\.venv\Scripts\python.exe src\plot_dataset_insights.py
```

Ejecutar análisis de actividades de walking:

```powershell
.\.venv\Scripts\python.exe src\plot_walking_triplet_insights.py
```

Entrenar baseline binario:

```powershell
.\.venv\Scripts\python.exe src\train_baseline.py
```

Entrenar XGBoost 6x6:

```powershell
.\.venv\Scripts\python.exe src\train_full_xgboost.py
```

## Próximos pasos

- comparar modelos lineales vs XGBoost con las mismas particiones
- revisar importancia de variables en el modelo completo
- probar selección de features específica para `WALKING_UPSTAIRS` vs `WALKING_DOWNSTAIRS`
- limpiar o consolidar scripts exploratorios antiguos si dejan de ser necesarios
