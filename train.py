import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "SMS.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_spam_pipeline.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")


def build_pipeline():
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=7000, sublinear_tf=True, min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=3000, sublinear_tf=True, min_df=2,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])

    svm = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    nb = MultinomialNB(alpha=0.3)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("nb", nb), ("lr", lr)],
        voting="soft", weights=[2, 1, 1], n_jobs=-1,
    )

    return ImbPipeline([
        ("features", features),
        ("smote", SMOTE(random_state=42)),
        ("clf", ensemble),
    ])


def load_dataset():
    """
    Load the SMS.csv dataset (teacher-provided file).
    Expected columns: v1 (label), v2 (message)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"[!] Dataset not found: {DATA_PATH}\n"
            f"    Please place the SMS.csv file inside the data/ folder."
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Handle both possible column names: v1/v2 or label/message
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    else:
        raise ValueError(
            f"[!] Unrecognized column names in CSV. Found: {list(df.columns)}"
        )

    df = df.dropna().drop_duplicates()

    # Map labels to 0 (ham) and 1 (spam)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Apply text cleaning
    df["clean_message"] = df["message"].apply(clean_text)
    df = df[df["clean_message"].str.len() > 0].reset_index(drop=True)

    print(f"[OK] Loaded {len(df)} rows from {DATA_PATH}")
    print(f"     ham={(df.label == 0).sum()}   spam={(df.label == 1).sum()}")
    return df


def train_model():
    df = load_dataset()

    X = df["clean_message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    print("[*] Training ensemble model (SVM + NB + LR) ...")
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall":    float(recall_score(y_test, pred)),
        "f1":        float(f1_score(y_test, pred)),
        "roc_auc":   float(roc_auc_score(y_test, proba)),
    }

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("\n" + classification_report(y_test, pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"\n5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Model saved   : {MODEL_PATH}")
    print(f"[OK] Metrics saved : {METRICS_PATH}")
    return pipe


if __name__ == "__main__":
    train_model()