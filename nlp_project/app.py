"""
Step 7: Deployment — Streamlit Web App
Language Identification System for Swahili, English, Sheng, and Luo
"""

import os
import pickle
import re
import string

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Language Identifier",
    page_icon="🌍",
    layout="centered",
)

# ── Language metadata ─────────────────────────────────────────────────────────
LANG_INFO = {
    "English": {
        "flag": "🇬🇧",
        "color": "#1565C0",
        "bg":    "#E3F2FD",
        "desc":  "Indo-European language; global lingua franca.",
        "example": "I am going to school today.",
    },
    "Kiswahili": {
        "flag": "🇰🇪",
        "color": "#2E7D32",
        "bg":    "#E8F5E9",
        "desc":  "Bantu language; official language of Kenya & Tanzania.",
        "example": "Ninakwenda shuleni leo.",
    },
    "Sheng": {
        "flag": "🏙️",
        "color": "#E65100",
        "bg":    "#FFF3E0",
        "desc":  "Nairobi urban slang mixing Swahili, English & local languages.",
        "example": "Niaje msee, mambo vipi?",
    },
    "Luo": {
        "flag": "🌊",
        "color": "#6A1B9A",
        "bg":    "#F3E5F5",
        "desc":  "Nilotic language spoken in western Kenya & Uganda.",
        "example": "An gi tich matek.",
    },
}


# ── Preprocessing (must mirror training) ─────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return " ".join(text.split())


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(model_dir, "best_model.pkl"),      "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "vectorizer.pkl"),      "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(model_dir, "labels.pkl"),          "rb") as f:
        labels = pickle.load(f)
    model_name = "Best Model"
    name_file = os.path.join(model_dir, "best_model_name.txt")
    if os.path.exists(name_file):
        with open(name_file) as f:
            model_name = f.read().strip()
    return model, vectorizer, labels, model_name


def predict(text, model, vectorizer, labels):
    clean = preprocess_text(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]

    # Confidence scores (probabilities where available)
    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidences = dict(zip(labels, probs))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        # Softmax-like normalisation for SVM scores
        exp_s = [2 ** s for s in scores]
        total = sum(exp_s)
        confidences = {lbl: v / total for lbl, v in zip(labels, exp_s)}
    return pred, confidences


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png",
             width=120)
    st.markdown("## 🌍 Language Identifier")
    st.markdown(
        "Automatically detects text written in **English**, **Kiswahili**, "
        "**Sheng**, or **Luo**."
    )
    st.divider()
    st.markdown("### Supported Languages")
    for lang, meta in LANG_INFO.items():
        st.markdown(
            f"<div style='background:{meta['bg']};border-left:4px solid {meta['color']};"
            f"padding:6px 10px;border-radius:4px;margin-bottom:6px;'>"
            f"<b>{meta['flag']} {lang}</b><br>"
            f"<small style='color:#555'>{meta['desc']}</small></div>",
            unsafe_allow_html=True
        )
    st.divider()

    # ── Model results (if available)
    results_path = os.path.join(os.path.dirname(__file__), "outputs", "model_results.csv")
    if os.path.exists(results_path):
        st.markdown("### 📊 Model Performance")
        df_r = pd.read_csv(results_path)
        st.dataframe(df_r.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.caption("CSC423 NLP Term Project · 2025")


# ── Main page ─────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Language Identification System")
st.markdown("**CSC423 NLP Term Project** — Kenyan Multilingual Text Classifier")
st.divider()

# Load model
try:
    model, vectorizer, labels, model_name = load_model()
    st.success(f"✅ Model loaded: **{model_name}**", icon="🤖")
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Please run `python train_evaluate.py` first "
        "to train and save the model.",
        icon="🚨"
    )
    st.stop()

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown("### ✍️ Enter Text to Identify")

user_input = st.text_area(
    label="Type or paste text below:",
    placeholder="e.g.  Niaje msee, mambo vipi?",
    height=120,
    key="text_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    identify_btn = st.button("🔍 Identify", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=False)

if clear_btn:
    st.session_state["text_input"] = ""
    st.rerun()

# ── Prediction ────────────────────────────────────────────────────────────────
if identify_btn:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing text …"):
            pred, confidences = predict(user_input, model, vectorizer, labels)

        meta = LANG_INFO.get(pred, {"flag": "🌐", "color": "#333", "bg": "#F5F5F5", "desc": ""})

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        st.markdown(
            f"""
            <div style="
                background:{meta['bg']};
                border-left: 6px solid {meta['color']};
                border-radius: 8px;
                padding: 20px 24px;
                margin-bottom: 12px;
            ">
                <div style="font-size:2.2rem; font-weight:800; color:{meta['color']};">
                    {meta['flag']}  {pred}
                </div>
                <div style="color:#555; margin-top:6px;">{meta['desc']}</div>
                <div style="margin-top:12px; background:#fff; padding:8px 12px;
                            border-radius:4px; font-style:italic; color:#333;">
                    <b>Your text:</b> {user_input}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence bar chart
        if confidences:
            st.markdown("#### Confidence Scores")
            fig, ax = plt.subplots(figsize=(6, 3))
            langs  = list(confidences.keys())
            scores = [confidences[l] * 100 for l in langs]
            colors = [LANG_INFO.get(l, {}).get("color", "#999") for l in langs]

            bars = ax.barh(langs, scores, color=colors, edgecolor="white", height=0.5)
            ax.set_xlim(0, 110)
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Model Confidence per Language", fontweight="bold")
            for bar, score in zip(bars, scores):
                ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
                        f"{score:.1f}%", va="center", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── Quick-test examples ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Quick Test Examples")

examples = [
    ("Kiswahili", "Mama anapika chakula kitamu sana jikoni."),
    ("English",   "The children are playing football in the park."),
    ("Sheng",     "Niaje msee, ile bash ilikuwa moto kabisa usiku."),
    ("Luo",       "An gi tich matek e ot."),
]

cols = st.columns(2)
for i, (lang, example) in enumerate(examples):
    meta = LANG_INFO.get(lang, {})
    with cols[i % 2]:
        if st.button(
            f"{meta.get('flag','')} {lang}\n\"{example[:40]}…\"" if len(example) > 40 else f"{meta.get('flag','')} {lang}\n\"{example}\"",
            use_container_width=True,
            key=f"example_{i}"
        ):
            pred, confidences = predict(example, model, vectorizer, labels)
            st.markdown(
                f"<div style='background:{meta['bg']};border-left:4px solid {meta['color']};"
                f"border-radius:6px;padding:10px 14px;'>"
                f"<b>Input:</b> {example}<br>"
                f"<b>Predicted:</b> {meta.get('flag','')} <b>{pred}</b></div>",
                unsafe_allow_html=True
            )

# ── Visual outputs ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Training Results & Visualisations")

tab1, tab2, tab3 = st.tabs(["Model Comparison", "Data Distribution", "Confusion Matrices"])

with tab1:
    p = os.path.join(os.path.dirname(__file__), "outputs", "model_comparison.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)
    else:
        st.info("Run `python train_evaluate.py` to generate charts.")

with tab2:
    p = os.path.join(os.path.dirname(__file__), "outputs", "data_distribution.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)
    else:
        st.info("Run `python train_evaluate.py` to generate charts.")

with tab3:
    for name_key in ["naive_bayes", "logistic_regression", "svm_linearSVC"]:
        # try both possible filename variants
        for variant in [name_key, name_key.replace("svm_linearSVC", "svm_linearsvc")]:
            p = os.path.join(
                os.path.dirname(__file__), "outputs",
                f"confusion_matrix_{variant}.png"
            )
            if os.path.exists(p):
                st.image(p, use_column_width=True)
                break
    # Also show any confusion matrix files present
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    if os.path.exists(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.startswith("confusion_matrix") and f.endswith(".png"):
                st.image(os.path.join(out_dir, f), use_column_width=True)
