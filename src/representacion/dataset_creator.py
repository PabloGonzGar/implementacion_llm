# ============================================================
# NOTEBOOK: dataset_creator.ipynb
# Propósito: Limpiar y construir el dataset compartido final
# Dataset origen: milistu/AMAZON-Products-2023 (HuggingFace)
# Output: dataset_marketing.parquet
# ============================================================

# ────────────────────────────────────────────────────────────
# 0. INSTALACIÓN Y LIBRERÍAS
# ────────────────────────────────────────────────────────────
# En Kaggle, datasets y pandas ya están disponibles.
# Solo necesitamos instalar la librería de HuggingFace si no está.

# !pip install datasets -q

import pandas as pd
import numpy as np
import re
import ast
from datasets import load_dataset

print("Librerías cargadas correctamente.")

# ────────────────────────────────────────────────────────────
# 1. CARGA DEL DATASET DESDE HUGGING FACE
# ────────────────────────────────────────────────────────────
# Cargamos el split "train" (el único disponible en este dataset).
# Lo convertimos a pandas inmediatamente para trabajar con comodidad.

print("Cargando dataset desde HuggingFace...")
raw = load_dataset("milistu/AMAZON-Products-2023", split="train")
df = raw.to_pandas()

print(f"Filas totales cargadas: {len(df):,}")
print(f"Columnas: {list(df.columns)}")

# ────────────────────────────────────────────────────────────
# 2. SELECCIÓN DE COLUMNAS RELEVANTES
# ────────────────────────────────────────────────────────────
# Eliminamos las columnas que no aportan a ninguna de las dos tareas
# (encoder de clasificación o decoder de generación de marketing).
# Las columnas eliminadas y su razón quedan documentadas aquí.

COLUMNAS_ELIMINAR = [
    "embeddings",          # Decidido previamente: no usamos embeddings precomputados
    "price",               # 31% de nulls, irrelevante para texto
    "store",               # Metadata de vendedor, sin valor predictivo
    "average_rating",      # Popularidad, no semántica
    "rating_number",       # Ídem
    "date_first_available",# Metadata temporal
    "parent_asin",         # Identificador interno
    "image",               # URLs de imágenes, no usamos visión
    "__index_level_0__",   # Artefacto de carga
]

# Eliminamos solo las columnas que realmente existen en el df
# (algunos artefactos pueden no estar siempre presentes)
columnas_a_eliminar = [c for c in COLUMNAS_ELIMINAR if c in df.columns]
df = df.drop(columns=columnas_a_eliminar)

print(f"\nColumnas tras selección: {list(df.columns)}")
print(f"Filas: {len(df):,}")

# ────────────────────────────────────────────────────────────
# 3. TABLA DE MAPEO: CATEGORÍAS ORIGINALES → MACRO-CLASES
# ────────────────────────────────────────────────────────────
# Esta tabla es la decisión central del proyecto.
# Cualquier categoría que no aparezca aquí → DROP.

CATEGORY_MAP = {
    # ── FASHION ──────────────────────────────────────────
    "AMAZON FASHION":                    "Fashion",

    # ── HOME & GARDEN ─────────────────────────────────────
    "Amazon Home":                       "Home & Garden",
    "Appliances":                        "Home & Garden",
    "Home Audio & Theater":              "Home & Garden",
    "Handmade":                          "Home & Garden",
    "Arts, Crafts & Sewing":             "Home & Garden",

    # ── ELECTRONICS & TECH ────────────────────────────────
    "All Electronics":                   "Electronics & Tech",
    "Cell Phones & Accessories":         "Electronics & Tech",
    "Computers":                         "Electronics & Tech",
    "Camera & Photo":                    "Electronics & Tech",
    "Car Electronics":                   "Electronics & Tech",
    "GPS & Navigation":                  "Electronics & Tech",
    "Portable Audio & Accessories":      "Electronics & Tech",
    "Appstore for Android":              "Electronics & Tech",

    # ── TOOLS & AUTOMOTIVE ────────────────────────────────
    "Tools & Home Improvement":          "Tools & Automotive",
    "Automotive":                        "Tools & Automotive",
    "Industrial & Scientific":           "Tools & Automotive",

    # ── SPORTS & HEALTH ───────────────────────────────────
    "Sports & Outdoors":                 "Sports & Health",
    "Health & Personal Care":            "Sports & Health",
    "All Beauty":                        "Sports & Health",

    # ── ENTERTAINMENT ─────────────────────────────────────
    "Digital Music":                     "Entertainment",
    "Video Games":                       "Entertainment",
    "Musical Instruments":               "Entertainment",
    "Collectibles & Fine Art":           "Entertainment",
    "Collectible Coins":                 "Entertainment",

    # ── OFFICE & LIFESTYLE ────────────────────────────────
    "Office Products":                   "Office & Lifestyle",
    "Pet Supplies":                      "Office & Lifestyle",
    "Toys & Games":                      "Office & Lifestyle",

    # ── DROP (masa crítica insuficiente) ──────────────────
    "Grocery":                           "DROP",
    "None":                              "DROP",
}

# Etiquetas numéricas (necesarias para el encoder)
LABEL_MAP = {
    "Fashion":              0,
    "Home & Garden":        1,
    "Electronics & Tech":   2,
    "Tools & Automotive":   3,
    "Sports & Health":      4,
    "Entertainment":        5,
    "Office & Lifestyle":   6,
}

# ────────────────────────────────────────────────────────────
# 4. RECUPERACIÓN DE CATEGORÍAS DESDE `categories`
# ────────────────────────────────────────────────────────────
# Hay 24,805 filas con main_category nula.
# Intentamos recuperar la categoría desde el primer elemento
# de la lista `categories`, que contiene la ruta jerárquica.

def get_category_from_list(cat_list):
    """
    Extrae el primer elemento de la lista de categorías del producto.
    Maneja el caso en que la columna llegue como string serializado.
    """
    # Si ya es una lista de Python, usarla directamente
    if isinstance(cat_list, list):
        return cat_list[0] if len(cat_list) > 0 else None

    # Si llegó como string (artefacto de serialización), parsearla
    if isinstance(cat_list, str):
        try:
            parsed = ast.literal_eval(cat_list)
            return parsed[0] if len(parsed) > 0 else None
        except (ValueError, SyntaxError):
            return None

    return None


# Rellenamos main_category nula con el primer elemento de categories
mask_null = df["main_category"].isna() | (df["main_category"] == "None")
print(f"\nFilas con main_category nula antes de recuperación: {mask_null.sum():,}")

df.loc[mask_null, "main_category"] = df.loc[mask_null, "categories"].apply(
    get_category_from_list
)

# Cuántas recuperamos
mask_null_post = df["main_category"].isna() | (df["main_category"] == "None")
recuperadas = mask_null.sum() - mask_null_post.sum()
print(f"Filas recuperadas desde `categories`: {recuperadas:,}")
print(f"Filas aún sin categoría (serán eliminadas): {mask_null_post.sum():,}")

# ────────────────────────────────────────────────────────────
# 5. APLICAR EL MAPEO DE MACRO-CLASES
# ────────────────────────────────────────────────────────────

df["main_category_clean"] = df["main_category"].map(CATEGORY_MAP)
df["label"] = df["main_category_clean"].map(LABEL_MAP)

# Eliminar filas cuya categoría no está en el mapa
# (incluyendo DROP y cualquier categoría nueva no contemplada)
filas_antes = len(df)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
filas_eliminadas = filas_antes - len(df)

print(f"\nFilas eliminadas tras mapeo (DROP + no mapeadas): {filas_eliminadas:,}")
print(f"Filas finales: {len(df):,}")
print(f"\nDistribución de macro-clases:")
print(df["main_category_clean"].value_counts())

# ────────────────────────────────────────────────────────────
# 6. LIMPIEZA DE `description` Y CONSTRUCCIÓN DE `text_input`
# ────────────────────────────────────────────────────────────

def clean_description(desc, title, max_chars=500):
    """
    Limpia la descripción del producto:
    1. Elimina artefactos de scraping ("Amazon.com" al inicio).
    2. Elimina el título del inicio de la descripción si está repetido.
    3. Normaliza espacios y saltos de línea.
    4. Trunca a max_chars para evitar textos enormes (tracklists, etc.).
    Devuelve string vacío si queda vacío tras la limpieza.
    """
    if not isinstance(desc, str) or desc.strip() == "":
        return ""

    # Normalizar saltos de línea y espacios múltiples
    desc = re.sub(r'[\r\n]+', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()

    # Eliminar "Amazon.com" al inicio (artefacto de scraping)
    desc = re.sub(r'^Amazon\.com\s*', '', desc, flags=re.IGNORECASE).strip()

    # Eliminar repetición del título al inicio de la descripción
    # Comparación case-insensitive, ignorando puntuación extra
    if isinstance(title, str) and len(title) > 10:
        title_clean = re.escape(title.strip())
        desc = re.sub(r'^' + title_clean + r'\s*', '', desc,
                      flags=re.IGNORECASE).strip()

    # Truncar a max_chars para evitar textos enormes
    if len(desc) > max_chars:
        # Truncar en el último espacio antes del límite para no cortar palabras
        desc = desc[:max_chars].rsplit(' ', 1)[0]

    return desc


def build_text_input(title, description_clean):
    """
    Construye el campo text_input para el encoder.
    Formato: "título [SEP] descripción" si hay descripción,
             "título" si no hay.
    """
    title = str(title).strip() if isinstance(title, str) else ""
    if description_clean and description_clean.strip():
        return f"{title} [SEP] {description_clean.strip()}"
    return title


print("\nLimpiando descriptions y construyendo text_input...")

df["description_clean"] = df.apply(
    lambda row: clean_description(row["description"], row["title"]),
    axis=1
)

df["text_input"] = df.apply(
    lambda row: build_text_input(row["title"], row["description_clean"]),
    axis=1
)

# Estadísticas rápidas de calidad
n_sin_desc = (df["description_clean"] == "").sum()
print(f"Productos sin description útil (usan solo título): {n_sin_desc:,} "
      f"({100 * n_sin_desc / len(df):.1f}%)")

longitudes = df["text_input"].str.len()
print(f"Longitud media de text_input (chars): {longitudes.mean():.0f}")
print(f"Percentil 50: {longitudes.quantile(0.5):.0f} chars")
print(f"Percentil 90: {longitudes.quantile(0.9):.0f} chars")
print(f"Percentil 99: {longitudes.quantile(0.99):.0f} chars")

# ────────────────────────────────────────────────────────────
# 7. VERIFICACIÓN DE COLUMNAS ANIDADAS (features, details, categories)
# ────────────────────────────────────────────────────────────
# Verificamos que estas columnas llegan como tipos Python nativos
# y no como strings serializados, que romperían el trabajo de Adrián.

def ensure_list(val):
    """Garantiza que el valor sea una lista Python."""
    if isinstance(val, list):
        return val
    # numpy arrays (como los que produce HuggingFace al convertir a pandas)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    return []

def ensure_dict(val):
    """Garantiza que el valor sea un diccionario Python."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return {}
    return {}

print("\nVerificando y normalizando columnas anidadas...")
df["categories"] = df["categories"].apply(ensure_list)
df["features"]   = df["features"].apply(ensure_list)
df["details"]    = df["details"].apply(ensure_dict)

# Estadísticas de presencia (útiles para el README de ambos)
print(f"Productos con features no vacías: "
      f"{(df['features'].apply(len) > 0).sum():,} "
      f"({100 * (df['features'].apply(len) > 0).mean():.1f}%)")
print(f"Productos con details no vacías: "
      f"{(df['details'].apply(len) > 0).sum():,} "
      f"({100 * (df['details'].apply(len) > 0).mean():.1f}%)")

# ────────────────────────────────────────────────────────────
# 8. SHUFFLE Y RESET DEL ÍNDICE
# ────────────────────────────────────────────────────────────
# El dataset original viene ordenado por categoría.
# Shuffleamos con semilla fija para reproducibilidad.

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nDataset shuffleado. Primeras categorías tras shuffle:")
print(df["main_category_clean"].head(10).values)

# ────────────────────────────────────────────────────────────
# 9. SELECCIÓN FINAL DE COLUMNAS Y ORDEN
# ────────────────────────────────────────────────────────────
# Definimos el orden lógico de columnas en el dataset compartido.
# Las primeras columnas son las más críticas para ambos.

COLUMNAS_FINALES = [
    # Columnas para el encoder (clasificación)
    "text_input",           # Input del encoder (título + descripción limpia)
    "label",                # Etiqueta numérica 0-6
    "main_category_clean",  # Nombre de la macro-clase (legible)

    # Columnas de texto raw (para encoder y decoder)
    "title",
    "description",          # Descripción original (Adrián puede querer limpiarla distinto)
    "description_clean",    # Versión limpia que ya usamos en text_input

    # Columnas estructuradas (core para el decoder)
    "features",
    "details",
    "categories",
]

# Mantener solo columnas que existen (por si alguna fue eliminada antes)
columnas_finales = [c for c in COLUMNAS_FINALES if c in df.columns]
df = df[columnas_finales]

print(f"\nColumnas del dataset final: {list(df.columns)}")
print(f"Shape final: {df.shape}")

# ────────────────────────────────────────────────────────────
# 10. EXPORTACIÓN
# ────────────────────────────────────────────────────────────

OUTPUT_PATH = "dataset_marketing.parquet"

df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

print(f"\n✅ Dataset exportado correctamente: {OUTPUT_PATH}")
print(f"   Filas: {len(df):,}")
print(f"   Columnas: {len(df.columns)}")
print(f"   Tamaño estimado en disco: ~{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

# Verificación de lectura
df_check = pd.read_parquet(OUTPUT_PATH)
assert len(df_check) == len(df), "ERROR: El archivo exportado no coincide en filas"
assert list(df_check.columns) == list(df.columns), "ERROR: Las columnas no coinciden"
print("   Verificación de lectura: ✅ OK")

# ────────────────────────────────────────────────────────────
# 11. RESUMEN FINAL PARA EL README
# ────────────────────────────────────────────────────────────
# Imprimimos los números que irán al README.

print("\n" + "="*60)
print("RESUMEN PARA EL README")
print("="*60)
print(f"Dataset origen: milistu/AMAZON-Products-2023 (HuggingFace)")
print(f"Filas en el origen: 117,243")
print(f"Filas eliminadas (main_category no mapeada / DROP): {117243 - len(df):,}")
print(f"Filas en el dataset final: {len(df):,}")
print(f"\nDistribución de clases:")
dist = df.groupby(["label", "main_category_clean"]).size().reset_index(name="n")
dist["pct"] = (dist["n"] / len(df) * 100).round(1)
print(dist.to_string(index=False))
print(f"\nProductos con descripción útil: "
      f"{(df['description_clean'] != '').sum():,} "
      f"({100 * (df['description_clean'] != '').mean():.1f}%)")