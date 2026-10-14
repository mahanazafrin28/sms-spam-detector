import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "SMS.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_spam_pipeline.pkl")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_dataframe():
    """Load and prepare the dataset for evaluation."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    else:
        raise ValueError(f"Unrecognized columns: {list(df.columns)}")

    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["clean_message"] = df["message"].apply(clean_text)
    df = df[df["clean_message"].str.len() > 0].reset_index(drop=True)
    return df


def generate_all_plots():
    pipe = joblib.load(MODEL_PATH)
    df = load_dataframe()

    _, X_test, _, y_test = train_test_split(
        df["clean_message"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"],
                yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # 3. Word Clouds
    for label, name in [(1, "spam"), (0, "ham")]:
        text = " ".join(df[df["label"] == label]["clean_message"])
        wc = WordCloud(width=900, height=450, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc)
        plt.axis("off")
        plt.title(f"WordCloud - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"wordcloud_{name}.png"), dpi=150)
        plt.close()

        print("\n===== Evaluation Metrics =====")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred):.4f}")
    print("==============================\n")
    
    print(f"[OK] Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    generate_all_plots()