"""
Step 3: Data Preprocessing
- Lowercasing
- Removing punctuation
- Tokenization
- Removing extra whitespace
- Sheng slang normalization
"""

import re
import string
import pandas as pd


# ── Sheng slang normalisation map ────────────────────────────────────────────
SHENG_SLANG = {
    "msee":   "mtu",
    "buda":   "rafiki",
    "demu":   "msichana",
    "chapaa": "pesa",
    "dough":  "pesa",
    "doh":    "pesa",
    "mob":    "sana",
    "base":   "nyumba",
    "rada":   "smart",
    "fit":    "sawa",
    "moto":   "nzuri",
    "fire":   "nzuri",
    "poa":    "sawa",
    "sawa":   "sawa",
    "hustle": "kazi",
    "bash":   "party",
    "beat":   "muziki",
    "game":   "mchezo",
    "connect":"mwunganisho",
    "deal":   "biashara",
    "count":  "hesabu",
    "sort":   "panga",
    "vibe":   "hali",
    "swag":   "mtindo",
    "broke":  "bila pesa",
}


def normalize_sheng(text: str) -> str:
    """Replace common Sheng slang words with normalised equivalents."""
    tokens = text.split()
    return " ".join(SHENG_SLANG.get(t, t) for t in tokens)


def preprocess_text(text: str, normalize_slang: bool = False) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove digits
    5. Collapse extra whitespace
    6. (Optional) Sheng slang normalisation
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Remove digits
    text = re.sub(r"\d+", "", text)

    # 5. Collapse whitespace
    text = " ".join(text.split())

    # 6. Slang normalisation
    if normalize_slang:
        text = normalize_sheng(text)

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the whole dataset.
    Adds a `clean_text` column and returns the updated dataframe.
    """
    print("\n" + "=" * 55)
    print("  STEP 3: DATA PREPROCESSING")
    print("=" * 55)

    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Drop rows where clean text is empty after preprocessing
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    after = len(df)

    if before != after:
        print(f"  Dropped {before - after} empty rows after preprocessing.")

    print(f"  Preprocessed {after} rows successfully.")
    print("\n  Sample (original → cleaned):")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"    [{row['language']}]")
        print(f"      Original : {row['text']}")
        print(f"      Cleaned  : {row['clean_text']}")

    return df


if __name__ == "__main__":
    from prepare_dataset import prepare_dataset
    df = prepare_dataset()
    df = preprocess_dataframe(df)
    df.to_csv("dataset_preprocessed.csv", index=False)
    print("\n✓ Saved → dataset_preprocessed.csv")
