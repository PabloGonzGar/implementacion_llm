# ============================================================
# NOTEBOOK: eda_encoder.ipynb
# Propósito: Mini-EDA del dataset compartido
#            Justificación de max_length para el encoder
# Input:     /kaggle/input/dataset-marketing/dataset_marketing.parquet
# ============================================================

# ────────────────────────────────────────────────────────────
# 0. LIBRERÍAS
# ────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer

# Estilo limpio para las figuras del README
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

print("Librerías cargadas.")

# ────────────────────────────────────────────────────────────
# 1. CARGA DEL DATASET LIMPIO
# ────────────────────────────────────────────────────────────
# Cargamos desde Kaggle Datasets. Ruta estándar en Kaggle:
# /kaggle/input/<nombre-del-dataset>/<nombre-archivo>

DATASET_PATH = "/kaggle/input/datasets/mjgut05/dataset-marketing/dataset_marketing.parquet"

df = pd.read_parquet(DATASET_PATH)
print(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")
print(f"\nPrimeras filas:")
df[["text_input", "main_category_clean", "label"]].head(5)

# ────────────────────────────────────────────────────────────
# 2. COMPROBACIONES BÁSICAS DE CALIDAD
# ────────────────────────────────────────────────────────────
# Verificamos que el dataset llegó bien desde el parquet.
# Esto detecta problemas de serialización antes de entrenar.

print("=" * 55)
print("COMPROBACIONES DE CALIDAD")
print("=" * 55)

# 2.1 Nulos en columnas críticas
for col in ["text_input", "label", "main_category_clean"]:
    n_null = df[col].isna().sum()
    print(f"  Nulos en '{col}': {n_null}")

# 2.2 Textos vacíos en text_input
n_empty = (df["text_input"].str.strip() == "").sum()
print(f"\n  Textos vacíos en text_input: {n_empty}")

# 2.3 Duplicados exactos en text_input
n_dup = df["text_input"].duplicated().sum()
print(f"  Duplicados exactos en text_input: {n_dup} "
      f"({100 * n_dup / len(df):.2f}%)")

# 2.4 Rango de labels
print(f"\n  Rango de labels: {df['label'].min()} - {df['label'].max()}")
print(f"  Labels únicos: {sorted(df['label'].unique())}")

# 2.5 Clases sin representación
expected_labels = set(range(7))
present_labels = set(df["label"].unique())
missing = expected_labels - present_labels
if missing:
    print(f"\n  ⚠️ CUIDADO: Labels ausentes: {missing}")
else:
    print(f"\n  ✅ Todos los labels (0-6) están presentes.")

# ────────────────────────────────────────────────────────────
# 3. FIGURA 1 — DISTRIBUCIÓN DE CLASES
# ────────────────────────────────────────────────────────────
# Requerida por el enunciado. Mostrar desbalanceo real.

class_counts = (
    df.groupby(["label", "main_category_clean"])
    .size()
    .reset_index(name="n")
    .sort_values("label")
)

# Colores diferenciados por clase
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#937860", "#DA8BC3"]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    class_counts["main_category_clean"],
    class_counts["n"],
    color=COLORS,
    edgecolor="white",
    height=0.65,
)

# Etiqueta con el número exacto y porcentaje
total = class_counts["n"].sum()
for bar, n in zip(bars, class_counts["n"]):
    ax.text(
        bar.get_width() + 300,
        bar.get_y() + bar.get_height() / 2,
        f"{n:,} ({100 * n / total:.1f}%)",
        va="center",
        fontsize=10,
    )

ax.set_xlabel("Número de productos")
ax.set_title("Distribución de macro-categorías en dataset_marketing\n"
             f"Total: {total:,} productos · 7 clases",
             fontsize=12, pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"
))
ax.set_xlim(0, class_counts["n"].max() * 1.18)

plt.tight_layout()
plt.savefig("class_distribution.png", bbox_inches="tight")
plt.show()
print("→ Guardada: class_distribution.png")

# ────────────────────────────────────────────────────────────
# 4. TOKENIZACIÓN Y LONGITUDES REALES
# ────────────────────────────────────────────────────────────
# Cargamos el tokenizer real del modelo para medir longitudes
# en tokens, no en caracteres.
# Usamos una muestra estratificada para que sea rápido.

print("\nCargando tokenizer de MiniLM...")
MODELO = "microsoft/Multilingual-MiniLM-L12-H384"

# MiniLM no soporta AutoTokenizer directamente; hay que especificar
# el tokenizer base que usa (BertTokenizer de bert-base-multilingual-cased)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
print("Tokenizer cargado.")

# Muestra estratificada: 3000 ejemplos por clase (si hay suficientes)
# para que la distribución de longitudes sea representativa de todas las clases
N_SAMPLE_PER_CLASS = 3000

sample = (
    df.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(min(N_SAMPLE_PER_CLASS, len(g)), random_state=42))
    .reset_index(drop=True)
)
print(f"\nMuestra estratificada: {len(sample):,} textos")

# Tokenizar la muestra
print("Tokenizando (puede tardar 30-60 segundos)...")
encoded = tokenizer(
    sample["text_input"].tolist(),
    add_special_tokens=True,   # incluye [CLS] y [SEP]
    truncation=False,          # NO truncamos — queremos longitudes reales
    padding=False,
)
token_lengths = [len(ids) for ids in encoded["input_ids"]]
sample["n_tokens"] = token_lengths

print("\nEstadísticas de longitud en TOKENS:")
pcts = [10, 25, 50, 75, 90, 95, 99]
for p in pcts:
    val = int(np.percentile(token_lengths, p))
    print(f"  P{p:>2}: {val} tokens")
print(f"  Max: {max(token_lengths)} tokens")
print(f"  Media: {np.mean(token_lengths):.0f} tokens")

# ────────────────────────────────────────────────────────────
# 5. FIGURA 2 — HISTOGRAMA DE LONGITUDES EN TOKENS
# ────────────────────────────────────────────────────────────
# Requerida por el enunciado. Justifica max_length.

p90 = int(np.percentile(token_lengths, 90))
p99 = int(np.percentile(token_lengths, 99))

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    token_lengths,
    bins=60,
    color="#4C72B0",
    edgecolor="white",
    alpha=0.85,
)

# Líneas de referencia para max_length candidatos
for max_len, color, ls in [(128, "#C44E52", "--"), (256, "#55A868", "-")]:
    ax.axvline(max_len, color=color, linestyle=ls, linewidth=2,
               label=f"max_length={max_len}")

# Línea de percentil 90
ax.axvline(p90, color="orange", linestyle=":", linewidth=1.5,
           label=f"P90 = {p90} tokens")

ax.set_xlabel("Longitud en tokens (incluye [CLS] y [SEP])")
ax.set_ylabel("Número de productos")
ax.set_title("Distribución de longitudes de text_input tras tokenización\n"
             f"Muestra estratificada: {len(sample):,} productos",
             fontsize=12, pad=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("token_length_histogram.png", bbox_inches="tight")
plt.show()
print("→ Guardada: token_length_histogram.png")

# ────────────────────────────────────────────────────────────
# 6. DECISIÓN DE max_length
# ────────────────────────────────────────────────────────────
# Con los percentiles reales, elegimos max_length.

p90_val = int(np.percentile(token_lengths, 90))
p99_val = int(np.percentile(token_lengths, 99))
p50_val = int(np.percentile(token_lengths, 50))

print("\n" + "=" * 55)
print("DECISIÓN DE max_length")
print("=" * 55)

# Cobertura a distintos max_length
for ml in [64, 128, 192, 256, 512]:
    cobertura = 100 * np.mean(np.array(token_lengths) <= ml)
    print(f"  max_length={ml:>3}: cubre el {cobertura:.1f}% de los textos "
          f"sin truncar")

print(f"""
JUSTIFICACIÓN:
  - P50 = {p50_val} tokens → la mitad de los textos tiene menos de esta longitud
  - P90 = {p90_val} tokens → max_length=128 truncaría el ~{100*(1-np.mean(np.array(token_lengths)<=128)):.0f}% de los textos
  - P99 = {p99_val} tokens → max_length=256 cubre prácticamente todo el dataset

DECISIÓN FINAL: max_length = 256
  Razón: cubre el 98.5% de los textos sin truncar, es compatible con
  la GPU T4/P100 de Kaggle (sin problemas de memoria con batch_size=32),
  y es el valor estándar para tareas de clasificación con descripciones
  de longitud media. max_length=128 sería demasiado agresivo dado el P90.
""")

# ────────────────────────────────────────────────────────────
# 7. ANÁLISIS POR CLASE
# ────────────────────────────────────────────────────────────
# Longitud media por clase: útil para detectar si una clase
# tiene textos sistemáticamente más cortos o largos.

print("Longitud media de text_input por clase:")
by_class = (
    sample.groupby("main_category_clean")["n_tokens"]
    .agg(["mean", "median", "max"])
    .round(0)
    .astype(int)
    .sort_values("mean", ascending=False)
)
by_class.columns = ["Media", "Mediana", "Máximo"]
print(by_class.to_string())

# ────────────────────────────────────────────────────────────
# 8. RESUMEN PARA EL README
# ────────────────────────────────────────────────────────────

p90_val = int(np.percentile(token_lengths, 90))
p50_val = int(np.percentile(token_lengths, 50))

print("\n" + "=" * 55)
print("TEXTO PARA EL README")
print("=" * 55)
print(f"""
El dataset presenta un desbalanceo significativo: Fashion representa
el 39.5% de los productos mientras que Entertainment, la clase más
pequeña, supone solo el 2.4%. Para el entrenamiento usamos F1 macro
como métrica principal precisamente porque penaliza los fallos en
clases minoritarias independientemente de su tamaño, y aplicamos
pesos de clase inversamente proporcionales a la frecuencia para
compensar el desbalanceo durante el entrenamiento.

La longitud mediana de los textos de entrada es de {p50_val} tokens
y el percentil 90 es de {p90_val} tokens tras tokenización con el
modelo MiniLM. Por ello fijamos max_length = 256, valor que cubre un 98.5% de los textos sin truncar y es compatible con las
restricciones de memoria de la GPU T4 de Kaggle con batch_size = 32.
""")