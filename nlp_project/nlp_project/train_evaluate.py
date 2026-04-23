"""
Steps 4-6: Feature Extraction, Model Training & Evaluation
- TF-IDF + Character n-grams (very effective for language ID)
- Three models: Naive Bayes, Logistic Regression, SVM
- Full evaluation: accuracy, precision, recall, F1, confusion matrix
- Saves best model + vectorizer for deployment
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

from prepare_dataset import prepare_dataset
from preprocessing import preprocess_dataframe

warnings.filterwarnings("ignore")

# ── Output dirs ──────────────────────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Feature Extraction ───────────────────────────────────────────────────────
def build_vectorizer():
    """
    Combined word + character n-gram TF-IDF vectorizer.
    Character n-grams (2-4) are extremely effective for language identification
    because they capture morphological patterns unique to each language.
    """
    return TfidfVectorizer(
        analyzer="char_wb",   # character n-grams within word boundaries
        ngram_range=(2, 4),   # bigrams to 4-grams
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


# ── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=5.0, solver="lbfgs"
    ),
    "SVM (LinearSVC)": LinearSVC(C=1.0, max_iter=2000),
}


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test, labels, results_list):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results_list.append({
        "Model": name,
        "Accuracy":  round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec * 100, 2),
        "F1-Score":  round(f1 * 100, 2),
    })

    print(f"\n  {'─'*48}")
    print(f"  Model : {name}")
    print(f"  {'─'*48}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"outputs/confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix → outputs/confusion_matrix_{fname}.png")

    return y_pred


def plot_comparison(results):
    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(df))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[metric], width, label=metric, color=color, alpha=0.87)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — Language Identification", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Model"], fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        for j, val in enumerate(df[metric]):
            ax.text(j + i * width, val + 1, f"{val:.1f}", ha="center",
                    va="bottom", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    print("\n✓ Model comparison chart → outputs/model_comparison.png")


def plot_label_distribution(df):
    counts = df["language"].value_counts()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(counts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts.plot(kind="bar", ax=ax1, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Dataset Distribution by Language", fontweight="bold")
    ax1.set_xlabel("Language")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=30)
    for p in ax1.patches:
        ax1.annotate(str(int(p.get_height())),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=10)

    # Pie chart
    ax2.pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=140)
    ax2.set_title("Language Share (%)", fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/data_distribution.png", dpi=150)
    plt.close()
    print("✓ Data distribution chart → outputs/data_distribution.png")


# ── Main training pipeline ───────────────────────────────────────────────────
def train_and_evaluate():
    print("=" * 55)
    print("  STEPS 4-6: FEATURE EXTRACTION, TRAINING & EVALUATION")
    print("=" * 55)

    # ── Load & preprocess ────────────────────────────────────────────────────
    df_raw = prepare_dataset()
    df     = preprocess_dataframe(df_raw)

    labels = sorted(df["language"].unique().tolist())
    print(f"\n  Languages : {labels}")

    plot_label_distribution(df)

    # ── Train / test split ───────────────────────────────────────────────────
    # Convert to plain numpy arrays (avoid pyarrow StringArray indexing bug)
    X = df["clean_text"].to_numpy(dtype=str)
    y = df["language"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")

    # ── Vectorise ────────────────────────────────────────────────────────────
    print("\n  Building TF-IDF character n-gram vectorizer …")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Feature matrix shape : {X_train_vec.shape}")

    # ── Train & evaluate all models ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 55)

    results  = []
    trained  = {}
    best_f1  = 0.0
    best_name = None

    for name, clf in MODELS.items():
        print(f"\n  Training {name} …", end=" ", flush=True)
        clf.fit(X_train_vec, y_train)
        print("done.")

        evaluate_model(name, clf, X_test_vec, y_test, labels, results)
        trained[name] = clf

        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5, scoring="f1_weighted")
        print(f"  5-Fold CV F1 : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

        f1 = results[-1]["F1-Score"]
        if f1 > best_f1:
            best_f1  = f1
            best_name = name

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv("outputs/model_results.csv", index=False)
    print("\n✓ Results saved → outputs/model_results.csv")

    plot_comparison(results)

    # ── Save best model ───────────────────────────────────────────────────────
    best_model = trained[best_name]
    with open("models/best_model.pkl",  "wb") as f:
        pickle.dump(best_model, f)
    with open("models/vectorizer.pkl",  "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/labels.pkl",      "wb") as f:
        pickle.dump(labels, f)
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"\n✓ Best model : {best_name}  (F1 = {best_f1:.2f}%)")
    print("✓ Saved → models/best_model.pkl")
    print("✓ Saved → models/vectorizer.pkl")

    return best_model, vectorizer, labels


if __name__ == "__main__":
    train_and_evaluate()
