"""
pipeline.py — Clasificación + Generación de descripciones de marketing

Uso:
    python pipeline.py
    python pipeline.py --title "Producto" --features "feat1" "feat2"
"""

import argparse
import torch
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
)

# ── Rutas ──────────────────────────────────────────────────────────────────
ENCODER_CHECKPOINT = "models/encoder/checkpoint-4620"
DECODER_CHECKPOINT = "models/decoder/outputs/gpt2_marketing_completo/checkpoint_epoca_3"
ENCODER_TOKENIZER  = "bert-base-multilingual-cased"

# ── Labels ─────────────────────────────────────────────────────────────────
LABELS = {
    0: "Fashion",
    1: "Home & Garden",
    2: "Electronics & Tech",
    3: "Tools & Automotive",
    4: "Sports & Health",
    5: "Entertainment",
    6: "Office & Lifestyle",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    tokenizer = BertTokenizerFast.from_pretrained(ENCODER_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER_CHECKPOINT, num_labels=7
    )
    model.to(device).eval()
    return tokenizer, model


def load_decoder():
    tokenizer = AutoTokenizer.from_pretrained(DECODER_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(DECODER_CHECKPOINT)
    model.to(device).eval()
    return tokenizer, model


def generate_description(title, features, details, tok_dec, mod_dec, max_new_tokens=100):
    if features:
        ficha = " | ".join(features)
    elif details:
        ficha = " | ".join(f"{k}={v}" for k, v in details.items())
    else:
        ficha = title

    prompt = f"Input: {ficha}\nOutput:"
    encoding = tok_dec(prompt, return_tensors="pt")
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    with torch.no_grad():
        output_ids = mod_dec.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tok_dec.eos_token_id,
        )

    generated = tok_dec.decode(output_ids[0], skip_special_tokens=True)
    return generated.split("Output:")[-1].strip()


def classify(title, description, tok_enc, mod_enc):
    text = f"{title} [SEP] {description}"
    enc = tok_enc(
        text,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = mod_enc(**enc).logits

    label_id = logits.argmax().item()
    return LABELS[label_id], label_id


def run_pipeline(title, features=None, details=None):
    """
    Carga modelos, genera descripción y predice categoría.
    Retorna dict con description y category.
    """
    tok_enc, mod_enc = load_encoder()
    tok_dec, mod_dec = load_decoder()

    description = generate_description(
        title, features or [], details or {}, tok_dec, mod_dec
    )
    category, label_id = classify(title, description, tok_enc, mod_enc)

    return {
        "title": title,
        "generated_description": description,
        "category": category,
        "label": label_id,
    }


# ── Ejemplos demo ──────────────────────────────────────────────────────────
DEMO_PRODUCTS = [
    {
        "title": "Men's Running Shoes Lightweight Breathable",
        "features": [
            "Lightweight mesh upper for breathability",
            "Cushioned EVA midsole for impact absorption",
            "Non-slip rubber outsole",
            "Available in sizes 7-13",
            "Weight: 280g per shoe",
        ],
        "details": {},
    },
    {
        "title": "Wireless Bluetooth Noise Cancelling Headphones",
        "features": [
            "Active Noise Cancellation (ANC)",
            "40-hour battery life",
            "Bluetooth 5.0",
            "Foldable design",
            "Driver size: 40mm",
        ],
        "details": {},
    },
    {
        "title": "Cordless Electric Drill Set",
        "features": [
            "18V lithium-ion battery",
            "2-speed gearbox: 0-450 / 0-1500 RPM",
            "13mm keyless chuck",
            "Includes 25 drill bits",
            "LED work light",
        ],
        "details": {},
    },
]


def main():
    parser = argparse.ArgumentParser(description="Pipeline encoder+decoder productos")
    parser.add_argument("--title", type=str, help="Título del producto")
    parser.add_argument("--features", nargs="+", help="Características técnicas")
    args = parser.parse_args()

    print(f"Dispositivo: {device}\n")

    if args.title:
        result = run_pipeline(args.title, features=args.features or [])
        print(f"Título:      {result['title']}")
        print(f"Descripción: {result['generated_description']}")
        print(f"Categoría:   {result['category']} (label {result['label']})")
    else:
        tok_enc, mod_enc = load_encoder()
        tok_dec, mod_dec = load_decoder()

        for i, product in enumerate(DEMO_PRODUCTS, 1):
            print(f"{'='*60}")
            print(f"Ejemplo {i}")
            print(f"Título:      {product['title']}")

            description = generate_description(
                product["title"], product["features"], product["details"],
                tok_dec, mod_dec,
            )
            category, label_id = classify(product["title"], description, tok_enc, mod_enc)

            print(f"Encoder →    {category} (label {label_id})")
            print(f"Decoder →    {description}")
            print()


if __name__ == "__main__":
    main()
