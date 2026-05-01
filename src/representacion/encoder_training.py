# ============================================================
# NOTEBOOK: encoder_training.ipynb
# Propósito: Fine-tuning de MiniLM para clasificación de productos
# Input:     /kaggle/input/dataset-marketing/dataset_marketing.parquet
# Output:    /kaggle/working/encoder_checkpoint/
# ============================================================

# ────────────────────────────────────────────────────────────
# 0. INSTALACIÓN Y LIBRERÍAS
# ────────────────────────────────────────────────────────────

# !pip install transformers datasets accelerate -q

import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Semilla global para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Comprobación de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN CENTRAL
# ────────────────────────────────────────────────────────────
# Todos los hiperparámetros en un solo lugar.
# Si hay que ajustar algo, se cambia aquí y punto.

CFG = {
    # Modelo
    "model_name":       "microsoft/Multilingual-MiniLM-L12-H384",
    "tokenizer_name":   "bert-base-multilingual-cased",  # tokenizer real de MiniLM
    "num_labels":       7,
    "max_length":       256,   # justificado en EDA: cubre el 98.5% sin truncar

    # Dataset
    "dataset_path":     "/kaggle/input/datasets/mjgut05/dataset-marketing/dataset_marketing.parquet",
    "val_size":         0.10,  # 10% validación
    "test_size":        0.10,  # 10% test (evaluación final)

    # Entrenamiento
    "batch_size":       32,    # por dispositivo; con fp16 cabe en T4 16GB
    "epochs":           4,     # conservador; early stopping lo ajustará
    "learning_rate":    2e-5,  # estándar para fine-tuning de BERT-family
    "weight_decay":     0.01,
    "warmup_ratio":     0.1,   # 10% del total de steps para warmup
    "fp16":             True,  # mixed precision — gratis en T4

    # DataLoader
    "num_workers":      2,     # precarga de batches en paralelo

    # Output
    "output_dir":       "/kaggle/working/encoder_checkpoint",
    "logging_steps":    50,
    "eval_strategy":    "epoch",
    "save_strategy":    "epoch",
    "load_best_model":  True,
    "metric_for_best":  "eval_f1_macro",
    "early_stop_patience": 2,
}

# Mapas de label para el README y análisis de errores
LABEL2NAME = {
    0: "Fashion",
    1: "Home & Garden",
    2: "Electronics & Tech",
    3: "Tools & Automotive",
    4: "Sports & Health",
    5: "Entertainment",
    6: "Office & Lifestyle",
}

print("Configuración cargada.")
print(f"  max_length: {CFG['max_length']}")
print(f"  batch_size: {CFG['batch_size']}")
print(f"  epochs:     {CFG['epochs']}")
print(f"  fp16:       {CFG['fp16']}")

# ────────────────────────────────────────────────────────────
# 2. CARGA Y PARTICIÓN DEL DATASET
# ────────────────────────────────────────────────────────────

print("\nCargando dataset...")
df = pd.read_parquet(CFG["dataset_path"])
print(f"Cargado: {len(df):,} filas")

# Nos quedamos solo con lo que necesita el encoder
df_enc = df[["text_input", "label"]].copy()
df_enc["label"] = df_enc["label"].astype(int)

# Partición estratificada: train / val / test
# Estratificada = respeta las proporciones de clase en cada split
df_train, df_temp = train_test_split(
    df_enc,
    test_size=CFG["val_size"] + CFG["test_size"],
    stratify=df_enc["label"],
    random_state=SEED,
)
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.5,
    stratify=df_temp["label"],
    random_state=SEED,
)

print(f"\nSplits:")
print(f"  Train: {len(df_train):,} ({100*len(df_train)/len(df_enc):.0f}%)")
print(f"  Val:   {len(df_val):,}   ({100*len(df_val)/len(df_enc):.0f}%)")
print(f"  Test:  {len(df_test):,}   ({100*len(df_test)/len(df_enc):.0f}%)")

# Verificar que la distribución de clases es similar en cada split
print("\nDistribución de clases por split (%):")
for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    dist = split["label"].value_counts(normalize=True).sort_index()
    print(f"  {name}: " + " | ".join(
        f"{LABEL2NAME[i][:6]}: {v*100:.1f}%" for i, v in dist.items()
    ))

# ────────────────────────────────────────────────────────────
# 3. TOKENIZACIÓN
# ────────────────────────────────────────────────────────────
# Tokenizamos todo antes de entrenar. Así cada época
# solo trabaja sobre tensores ya preparados.

print("\nCargando tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained(CFG["tokenizer_name"])
print("Tokenizer listo.")

def tokenize_batch(batch):
    return tokenizer(
        batch["text_input"],
        max_length=CFG["max_length"],
        truncation=True,
        padding="max_length",  # padding fijo — más eficiente en Kaggle
    )

# Convertir a datasets.Dataset y tokenizar
print("Tokenizando splits (esto puede tardar 3-5 minutos en total)...")
for split_name, df_split in [
    ("train", df_train), ("val", df_val), ("test", df_test)
]:
    print(f"  Tokenizando {split_name}...")

ds = {
    split_name: (
        Dataset.from_pandas(df_split, preserve_index=False)
        .map(tokenize_batch, batched=True, batch_size=512,
             num_proc=2,   # paralelización de la tokenización
             remove_columns=["text_input"])
        .rename_column("label", "labels")
        .with_format("torch")
    )
    for split_name, df_split in [
        ("train", df_train), ("val", df_val), ("test", df_test)
    ]
}

print("Tokenización completada.")
print(f"  Columnas del dataset tokenizado: {ds['train'].column_names}")

# ────────────────────────────────────────────────────────────
# 4. PESOS DE CLASE (para compensar el desbalanceo)
# ────────────────────────────────────────────────────────────
# Fashion = 39.5% del dataset. Sin pesos de clase, el modelo
# puede aprender a predecir Fashion casi siempre y tener
# accuracy alta pero F1 macro pésimo.

label_counts = df_train["label"].value_counts().sort_index()
n_total = len(df_train)
n_classes = CFG["num_labels"]

# Peso inversamente proporcional a la frecuencia de cada clase
# La normalización * n_classes mantiene el gradiente en la misma escala
class_weights = (n_total / (n_classes * label_counts)).values
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\nPesos de clase calculados:")
for label, weight in enumerate(class_weights):
    print(f"  {LABEL2NAME[label]:<22}: {weight:.3f}")

# ────────────────────────────────────────────────────────────
# 5. MODELO
# ────────────────────────────────────────────────────────────

print(f"\nCargando modelo {CFG['model_name']}...")
model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"],
    num_labels=CFG["num_labels"],
    ignore_mismatched_sizes=True,  # la cabeza de clasificación es nueva
)

# Contar parámetros
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros totales:     {n_params:,}")
print(f"Parámetros entrenables: {n_trainable:,}")

# ────────────────────────────────────────────────────────────
# 6. TRAINER CON WEIGHTED LOSS
# ────────────────────────────────────────────────────────────
# Subclasificamos Trainer solo para sobreescribir compute_loss
# e inyectar los pesos de clase. Nada más cambia.

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ────────────────────────────────────────────────────────────
# 7. MÉTRICAS
# ────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":   f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy":   accuracy_score(labels, preds),
    }

# ────────────────────────────────────────────────────────────
# 8. TRAINING ARGUMENTS
# ────────────────────────────────────────────────────────────

total_train_steps = (len(ds["train"]) // CFG["batch_size"]) * CFG["epochs"]
warmup_steps = int(total_train_steps * CFG["warmup_ratio"])

print(f"\nSteps totales de entrenamiento: {total_train_steps:,}")
print(f"Warmup steps: {warmup_steps}")

training_args = TrainingArguments(
    output_dir=CFG["output_dir"],

    # Entrenamiento
    num_train_epochs=CFG["epochs"],
    per_device_train_batch_size=CFG["batch_size"],
    per_device_eval_batch_size=CFG["batch_size"] * 2,  # eval puede ser mayor
    learning_rate=CFG["learning_rate"],
    weight_decay=CFG["weight_decay"],
    warmup_steps=warmup_steps,

    # Precisión y velocidad
    fp16=CFG["fp16"],
    dataloader_num_workers=CFG["num_workers"],

    # Evaluación y checkpoints
    eval_strategy=CFG["eval_strategy"],
    save_strategy=CFG["save_strategy"],
    load_best_model_at_end=CFG["load_best_model"],
    metric_for_best_model=CFG["metric_for_best"],
    greater_is_better=True,
    save_total_limit=2,      # guardar solo los 2 mejores checkpoints

    # Logging
    logging_dir=os.path.join(CFG["output_dir"], "logs"),
    logging_steps=CFG["logging_steps"],
    report_to="none",        # sin WandB ni otras integraciones

    # Reproducibilidad
    seed=SEED,
)

# ────────────────────────────────────────────────────────────
# 9. ENTRENAMIENTO
# ────────────────────────────────────────────────────────────

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=CFG["early_stop_patience"]
        )
    ],
)

print("\n" + "=" * 55)
print("INICIANDO ENTRENAMIENTO")
print("=" * 55)
train_result = trainer.train()

print(f"\nEntrenamiento finalizado.")
print(f"  Tiempo total: {train_result.metrics['train_runtime']:.0f}s "
      f"({train_result.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/s:    {train_result.metrics['train_samples_per_second']:.1f}")

# ────────────────────────────────────────────────────────────
# 10. EVALUACIÓN SOBRE TEST
# ────────────────────────────────────────────────────────────
# Solo se evalúa sobre test UNA VEZ, al final.
# No se toca test durante el desarrollo — solo aquí.

print("\n" + "=" * 55)
print("EVALUACIÓN FINAL SOBRE TEST")
print("=" * 55)

test_preds_output = trainer.predict(ds["test"])
test_logits = test_preds_output.predictions
test_labels = test_preds_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)

f1_macro  = f1_score(test_labels, test_preds, average="macro")
f1_weighted = f1_score(test_labels, test_preds, average="weighted")
acc = accuracy_score(test_labels, test_preds)

print(f"\nResultados en test:")
print(f"  F1 macro:    {f1_macro:.4f}")
print(f"  F1 weighted: {f1_weighted:.4f}")
print(f"  Accuracy:    {acc:.4f}")

# Reporte completo por clase
print("\nReporte por clase:")
print(classification_report(
    test_labels, test_preds,
    target_names=[LABEL2NAME[i] for i in range(7)]
))

# ────────────────────────────────────────────────────────────
# 11. MATRIZ DE CONFUSIÓN
# ────────────────────────────────────────────────────────────

cm = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalizada por fila

class_names = [LABEL2NAME[i] for i in range(7)]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    linewidths=0.5,
)
ax.set_xlabel("Predicho", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title(
    f"Matriz de confusión (normalizada por fila)\n"
    f"F1 macro = {f1_macro:.3f} · F1 weighted = {f1_weighted:.3f}",
    fontsize=12, pad=12
)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches="tight", dpi=150)
plt.show()
print("→ Guardada: confusion_matrix.png")

# ────────────────────────────────────────────────────────────
# 12. EXPORTACIÓN DEL CHECKPOINT
# ────────────────────────────────────────────────────────────
# Guardamos modelo + tokenizer en una carpeta limpia.
# Esta carpeta es el entregable final de esta fase.

EXPORT_PATH = "/kaggle/working/encoder_final"
os.makedirs(EXPORT_PATH, exist_ok=True)

trainer.save_model(EXPORT_PATH)
tokenizer.save_pretrained(EXPORT_PATH)

# Guardamos también el mapa de labels para que pipeline.py sepa decodificar
import json
with open(os.path.join(EXPORT_PATH, "label_map.json"), "w") as f:
    json.dump(LABEL2NAME, f, indent=2, ensure_ascii=False)

print(f"\n✅ Checkpoint exportado en: {EXPORT_PATH}")
print(f"   Contenido:")
for fname in sorted(os.listdir(EXPORT_PATH)):
    fsize = os.path.getsize(os.path.join(EXPORT_PATH, fname))
    print(f"   {fname:<35} {fsize/1e6:.1f} MB")

# ────────────────────────────────────────────────────────────
# 13. PRECAUCIONES
# ────────────────────────────────────────────────────────────
# Este bloque imprime un diagnóstico rápido para saber si
# el experimento fue bien o hay que revisar algo.

print("\n" + "=" * 55)
print("DIAGNÓSTICO DEL EXPERIMENTO")
print("=" * 55)

# Señal 1: F1 macro mínimo aceptable
if f1_macro >= 0.75:
    print(f"  ✅ F1 macro = {f1_macro:.3f} — resultado sólido")
elif f1_macro >= 0.60:
    print(f"  ⚠️  F1 macro = {f1_macro:.3f} — aceptable, revisar clases pequeñas")
else:
    print(f"  ❌ F1 macro = {f1_macro:.3f} — revisar pesos de clase y learning rate")

# Señal 2: Clase más problemática
per_class_f1 = f1_score(test_labels, test_preds, average=None)
worst_class = np.argmin(per_class_f1)
print(f"  Clase con peor F1: {LABEL2NAME[worst_class]} "
      f"(F1 = {per_class_f1[worst_class]:.3f})")

# Señal 3: Brecha train vs val (overfitting)
train_metrics = train_result.metrics
print(f"  Loss entrenamiento final: {train_metrics.get('train_loss', 'N/A'):.4f}")

import shutil, os

# Comprimir encoder_final en un zip dentro de /kaggle/working/
shutil.make_archive(
    "/kaggle/working/encoder_final",   # nombre del zip (sin extensión)
    "zip",
    "/kaggle/working/encoder_final"    # carpeta a comprimir
)

# Comprimir también el checkpoint
shutil.make_archive(
    "/kaggle/working/encoder_checkpoint_zip",
    "zip",
    "/kaggle/working/encoder_checkpoint"
)

print("Zips creados:")
for f in os.listdir("/kaggle/working/"):
    if f.endswith(".zip"):
        size = os.path.getsize(f"/kaggle/working/{f}") / 1e6
        print(f"  {f}: {size:.0f} MB")