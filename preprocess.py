import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# auto-download (safe if already present)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
# keep some negation words — important for spam/ham distinction
NEGATIONS = {"not", "no", "nor", "never", "none", "cannot", "cant", "dont", "wont"}
STOP_WORDS -= NEGATIONS

LEMM = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Advanced cleaning tuned for SMS spam."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # normalise currency & numbers BEFORE removing digits
    text = re.sub(r"£\s*\d+(?:\.\d+)?", " moneyamount ", text)
    text = re.sub(r"\$\s*\d+(?:\.\d+)?", " moneyamount ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumber ", text)   # phone / shortcodes
    text = re.sub(r"\b\d{1,2}\b", " shortnumber ", text)

    # URLs / emails / handles
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"@\w+", " mention ", text)

    # replace common sms abbreviations with normal words (helps TF-IDF)
    sms_map = {
        r"\bu\b": "you", r"\bur\b": "your", r"\bpls\b": "please",
        r"\bplz\b": "please", r"\bthx\b": "thanks", r"\bmsg\b": "message",
        r"\btxt\b": "text", r"\bfree\b": "free", r"\bwon\b": "win",
        r"\bwin\b": "win", r"\bcall\b": "call", r"\bclaim\b": "claim",
    }
    for pat, rep in sms_map.items():
        text = re.sub(pat, rep, text)

    # remove punctuation & extra spaces
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        LEMM.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 1
    ]
    return " ".join(tokens)