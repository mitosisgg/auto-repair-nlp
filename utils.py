import re
import pandas as pd

from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def get_stopwords(extra_keep: Optional[List[str]]=None, extra_stop: Optional[List[str]]=None) -> Tuple[set, set]:
    """Returns (stopwords, keepwords). keepwords are protected domain terms (never removed)."""
    stop = set(ENGLISH_STOP_WORDS)
    keep = set()
    domain_keep = {
        "not", "will", "oil","filter","engine","transmission","trans","brake","rotor","pad","pads","battery","coolant",
        "radiator","leak","hose","belt","alternator","starter","axle","tire","tires","alignment","sensor",
        "oxygen","o2","spark","plug","plugs","cylinder","gasket","pump","fuel","injector","injectors",
        "ac","a/c","air","compressor","clutch","shifter","steering","power","window","door","lock","locks"
    }
    keep |= domain_keep
    if extra_keep:
        keep |= set(w.lower() for w in extra_keep)
    if extra_stop:
        stop |= set(w.lower() for w in extra_stop)
    # Ensure keepwords are not in stopwords
    stop -= keep
    return stop, keep

def normalize_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""   
    # Lowercase, remove newlines, normalize whitespace, keep alphanumerics and punctuation
    t = text.lower()
    t = re.sub(r'\n+', ' ', t)  # Remove newlines
    t = re.sub(r'[0-9]+', '', t)  # remove all numbers
    t = re.sub(r'[^a-z\s\-\./]', ' ', t)  # keep hyphens, slashes, periods
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_texts(texts: List[str], nlp, stopwords: set, keepwords: set) -> str:
    processed = []
    if nlp is not None:
        for t in texts:
            t = normalize_text(safe_text(t))
            if not t:
                processed.append("")
                continue
            doc = nlp(t)
            toks = []
            for token in doc:
                if token.is_space or token.is_punct:
                    continue
                token_text = token.text.lower()
                if (token_text in stopwords) and (token_text not in keepwords):
                    continue
                if len(token_text) < 2 and (token_text not in keepwords):
                    continue
                toks.append(token_text)
            processed.append(" ".join(toks))
    else:
        # Simple regex-based tokenization fallback
        for t in texts:
            t = normalize_text(safe_text(t))
            toks = [w for w in t.split() if (w not in stopwords) or (w in keepwords)]
            processed.append(" ".join(toks))
    return " ".join(processed)

def normalize_headers(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True).str.rstrip('_').str.lstrip('_')
    return df