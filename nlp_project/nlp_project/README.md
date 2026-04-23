# CSC423 NLP Term Project
## Language Identification System for Short Text in Kenyan Languages

> Automatically detects whether text is written in **English**, **Kiswahili**, **Sheng**, or **Luo**.

---

## 📁 Project Structure

```
nlp_project/
│
├── Dataset.csv                  ← Original dataset (Kiswahili, English, Luo)
├── dataset_labeled.csv          ← Final labeled dataset (auto-generated, includes Sheng)
├── dataset_preprocessed.csv     ← Preprocessed dataset (auto-generated)
│
├── prepare_dataset.py           ← Step 1 & 2: Data collection & labelling
├── preprocessing.py             ← Step 3:     Data preprocessing
├── train_evaluate.py            ← Steps 4–6:  Feature extraction, training & evaluation
├── app.py                       ← Step 7:     Streamlit web app (deployment)
│
├── models/                      ← Saved model files (auto-generated)
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   ├── labels.pkl
│   └── best_model_name.txt
│
├── outputs/                     ← Charts & results (auto-generated)
│   ├── data_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix_naive_bayes.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_svm_linearsvc.png
│   └── model_results.csv
│
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Install Python
Make sure you have **Python 3.8 or higher** installed.
Download from: https://www.python.org/downloads/

### 2. Install Required Libraries

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs:
- `scikit-learn` — Machine learning models
- `pandas` / `numpy` — Data handling
- `matplotlib` / `seaborn` — Visualisation
- `streamlit` — Web app deployment

---

## 🚀 How to Run the Project

### Step 1 — (Optional) Preview Dataset Preparation
```bash
python prepare_dataset.py
```
- Loads the original `Dataset.csv`
- Adds 101 Sheng sentences to balance the dataset
- Saves `dataset_labeled.csv`

---

### Step 2 — Train the Models
```bash
python train_evaluate.py
```
This single command runs the **entire ML pipeline**:
1. Loads and prepares the dataset
2. Preprocesses text (lowercase, remove punctuation, tokenize)
3. Extracts features using **TF-IDF character n-grams (2–4)**
4. Trains **3 models**: Naive Bayes, Logistic Regression, SVM
5. Evaluates with accuracy, precision, recall, F1-score, confusion matrix
6. Saves the best model to `models/`
7. Saves all charts to `outputs/`

Expected output (approximate):
```
Model               Accuracy   F1-Score
Naive Bayes          99.06%    99.08%
Logistic Regression  99.37%    99.38%   ← Best model
SVM (LinearSVC)      99.37%    99.38%
```

---

### Step 3 — Launch the Web App
```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

#### Using the app:
1. Type or paste any text into the input box
2. Click **🔍 Identify**
3. See the predicted language with confidence scores
4. Use the **Quick Test Examples** buttons to try sample sentences

---

## 🗂️ Dataset Overview

| Language   | Samples | Source                          |
|------------|--------:|---------------------------------|
| Luo        |     600 | Original dataset                |
| English    |     499 | Original dataset                |
| Kiswahili  |     393 | Original dataset                |
| Sheng      |     101 | Manually collected (Nairobi)    |
| **Total**  | **1593**|                                 |

---

## 🧠 Feature Extraction

**TF-IDF Character N-grams (2–4)**

Character n-grams are the best feature for language identification because:
- They capture sub-word morphological patterns unique to each language
- They handle out-of-vocabulary words and slang (Sheng) well
- They work even on very short texts (1–2 sentences)

Example — the word `"anaenda"` produces character n-grams:
`an, na, ae, en, nd, da, ana, ane, nen, end, nda, ...`

---

## 📊 Model Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 99.06%   | 99.18%    | 99.06% | 99.08%   |
| Logistic Regression | 99.37%   | 99.43%    | 99.37% | 99.38%   |
| SVM (LinearSVC)     | 99.37%   | 99.43%    | 99.37% | 99.38%   |

> **Best Model**: Logistic Regression (saved automatically)

---

## 🔍 Preprocessing Steps

1. **Lowercase** — `"Niaje Msee"` → `"niaje msee"`
2. **Remove URLs** — strips any http/www links
3. **Remove punctuation** — `"Mambo?"` → `"Mambo"`
4. **Remove digits** — removes numbers
5. **Collapse whitespace** — normalises spacing
6. **Sheng normalisation** (optional) — maps slang terms to standard words

---

## 📝 Project Mark Breakdown

| Component                | Marks | Status      |
|--------------------------|-------|-------------|
| Data Collection          | 20    | ✅ Complete |
| Data Labelling           | 15    | ✅ Complete |
| Data Preprocessing       | 15    | ✅ Complete |
| Feature Extraction       | 5     | ✅ Complete |
| Model Training           | 5     | ✅ Complete |
| Model Evaluation         | —     | ✅ Complete |
| Deployment (Streamlit)   | 20    | ✅ Complete |
| **Total**                | **80**|             |

### Bonus Features Implemented
- ✅ Sheng (code-mixed Swahili + English slang) handling
- ✅ Slang normalisation dictionary
- ✅ Confidence score visualisation

---

## 🛠️ Technologies Used

- Python 3.8+
- scikit-learn (ML models, TF-IDF)
- Pandas / NumPy (data handling)
- Matplotlib / Seaborn (visualisation)
- Streamlit (web deployment)
