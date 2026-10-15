"""
Sentiment Analysis using Pretrained BERT and DistilBERT
======================================================

Inference only:
- Does NOT fine-tune or train either model.
- Uses already fine-tuned Hugging Face sentiment-classification checkpoints.
- Performs minimal CSV cleaning and small EDA.
- Runs predictions and compares the models on tweets with positive/negative labels.
- Neutral examples are reported separately because the selected checkpoints are
  binary SST-2 sentiment classifiers (positive/negative), not 3-class classifiers.

Dataset:
    Twitter_Data.csv
Expected columns:
    clean_text : tweet text
    category   : -1 = negative, 0 = neutral, 1 = positive

Run from lab3:
    python sentiment_bert_pretrained.py

Hugging Face pipeline documentation:
    https://huggingface.co/docs/transformers/en/main_classes/pipelines
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Twitter_Data.csv"
OUTPUT_DIR = BASE_DIR / "outputs_pretrained"
OUTPUT_DIR.mkdir(exist_ok=True)

# These are sentiment-classification checkpoints based on the BERT and
# DistilBERT architectures. They are already fine-tuned by their publishers.
MODELS = {
    "BERT": "textattack/bert-base-uncased-SST-2",
    "DistilBERT": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
}

# Keep this moderate for memory and runtime. Set to None to process all rows.
MAX_ROWS = 1000
BATCH_SIZE = 8

# Dataset labels: -1 negative, 0 neutral, 1 positive.
DATASET_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}


# ============================================================
# 2. DEVICE
# ============================================================
if torch.cuda.is_available():
    device = 0
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print("No CUDA GPU detected. Running inference on CPU.")

print("This script performs inference only; it does not train or fine-tune models.")


# ============================================================
# 3. LOAD DATA + MINIMAL CLEANING
# ============================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}\n"
        "Put Twitter_Data.csv in the same lab3 folder as this script."
    )

df = pd.read_csv(DATA_FILE)
print("\nOriginal dataset shape:", df.shape)
print("Columns found:", df.columns.tolist())

text_candidates = ["clean_text", "text", "tweet", "content"]
label_candidates = ["category", "sentiment", "label"]

text_col = next((col for col in text_candidates if col in df.columns), None)
label_col = next((col for col in label_candidates if col in df.columns), None)

if text_col is None or label_col is None:
    raise ValueError(
        "Could not identify text and label columns. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df[[text_col, label_col]].copy()
df.columns = ["text", "category"]

# Minimal, justified cleaning only.
df["text"] = df["text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
df["category"] = pd.to_numeric(df["category"], errors="coerce")
df = df.dropna(subset=["text", "category"])
df = df[df["text"].str.len() > 0]
df["category"] = df["category"].astype(int)
df = df[df["category"].isin(DATASET_LABELS)]
df = df.drop_duplicates(subset=["text", "category"]).reset_index(drop=True)
df["true_sentiment"] = df["category"].map(DATASET_LABELS)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS).copy()

print("\nShape after cleaning:", df.shape)
print("\nSentiment class counts:")
print(df["true_sentiment"].value_counts())

# Save cleaned data for reproducibility.
df.to_csv(OUTPUT_DIR / "cleaned_twitter_data.csv", index=False)


# ============================================================
# 4. MINIMAL EDA
# ============================================================
# Plot 1: class distribution.
class_counts = (
    df["true_sentiment"]
    .value_counts()
    .reindex(["negative", "neutral", "positive"], fill_value=0)
)

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_class_distribution.png", dpi=150)
plt.close()

# Plot 2: tweet length.
word_counts = df["text"].str.split().str.len()
print("\nTweet word-count summary:")
print(word_counts.describe().round(2))

plt.figure(figsize=(6, 4))
upper_limit = word_counts.quantile(0.99) if len(word_counts) else 0
word_counts.clip(upper=upper_limit).hist(bins=30)
plt.title("Tweet Length Distribution (word count)")
plt.xlabel("Words per tweet (capped at 99th percentile)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tweet_length_distribution.png", dpi=150)
plt.close()


# ============================================================
# 5. SELECT DATA THAT MATCHES THE MODELS' LABEL SPACE
# ============================================================
# Both selected checkpoints are binary SST-2 classifiers.
# Neutral rows cannot be fairly evaluated as a third class, so preserve them
# in the cleaned CSV but exclude them from binary accuracy/report calculations.
binary_df = df[df["category"].isin([-1, 1])].copy().reset_index(drop=True)
binary_df["expected_binary"] = binary_df["true_sentiment"]

print(f"\nRows in dataset: {len(df)}")
print(f"Positive/negative rows evaluated: {len(binary_df)}")
print(f"Neutral rows not included in binary evaluation: {(df['category'] == 0).sum()}")

if binary_df.empty:
    raise ValueError("No positive/negative examples remain for binary evaluation.")

# Limit GPU/CPU memory use by processing small chunks.
texts = binary_df["text"].astype(str).tolist()


# ============================================================
# 6. PRETRAINED MODEL INFERENCE + EVALUATION
# ============================================================
comparison_rows = []
all_predictions = binary_df[["text", "category", "true_sentiment"]].copy()

for model_label, model_name in MODELS.items():
    print("\n" + "=" * 68)
    print(f"Loading pretrained {model_label}: {model_name}")
    print("=" * 68)

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )

    predicted_labels = []
    confidence_scores = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        outputs = sentiment_pipeline(
            batch_texts,
            truncation=True,
            max_length=256,
            batch_size=BATCH_SIZE,
        )

        for result in outputs:
            # SST-2 checkpoints usually return LABEL_0/LABEL_1 or
            # NEGATIVE/POSITIVE. Normalize labels robustly.
            raw_label = str(result["label"]).upper()
            if raw_label in {"LABEL_1", "POSITIVE", "POS"}:
                normalized = "positive"
            elif raw_label in {"LABEL_0", "NEGATIVE", "NEG"}:
                normalized = "negative"
            else:
                raise ValueError(
                    f"Unexpected label '{result['label']}' from {model_label}. "
                    "Check this checkpoint's label mapping before evaluating."
                )

            predicted_labels.append(normalized)
            confidence_scores.append(float(result["score"]))

    true_labels = binary_df["expected_binary"].tolist()
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\n{model_label} binary accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=["negative", "positive"],
            target_names=["negative", "positive"],
            zero_division=0,
        )
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=["negative", "positive"],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"{model_label.lower()}_classification_report.csv"
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=["negative", "positive"],
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"],
    )
    display.plot(values_format="d")
    plt.title(f"{model_label} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_label.lower()}_confusion_matrix.png", dpi=150)
    plt.close()

    all_predictions[f"{model_label.lower()}_prediction"] = predicted_labels
    all_predictions[f"{model_label.lower()}_confidence"] = confidence_scores

    comparison_rows.append(
        {
            "model": model_label,
            "checkpoint": model_name,
            "accuracy": accuracy,
            "evaluated_examples": len(true_labels),
            "neutral_examples_excluded": int((df["category"] == 0).sum()),
        }
    )

    del sentiment_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 7. SAVE COMPARISON + SAMPLE OUTPUTS
# ============================================================
comparison = pd.DataFrame(comparison_rows).sort_values(
    "accuracy", ascending=False
)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
all_predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "mode": "inference_only_no_fine_tuning",
            "dataset_rows_used": int(len(df)),
            "binary_rows_evaluated": int(len(binary_df)),
            "neutral_rows_excluded_from_binary_evaluation": int((df["category"] == 0).sum()),
            "models": MODELS,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 68)
print("PRETRAINED MODEL COMPARISON")
print("=" * 68)
print(comparison.to_string(index=False))
print(f"\nSaved all outputs to: {OUTPUT_DIR.resolve()}")

print("\nSample predictions:")
print(all_predictions.head(10).to_string(index=False))

print(
    "\nImportant interpretation note: these are pretrained binary sentiment "
    "classifiers. Neutral tweets were not treated as positive or negative for "
    "the accuracy calculation. The comparison is on this dataset subset and "
    "may be affected by domain differences between SST-2 and Twitter text."
)
