"""
Steps:
  1. Lowercase
  2. Remove email addresses and URLs
  3. Keep only letters (remove digits and punctuation)
  4. Tokenize with NLTK
  5. Remove English stopwords
  6. Lemmatize with WordNet
  7. Discard tokens shorter than 2 chars
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only first time)
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove emails/URLs/non-letters, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)             # emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # keep letters/spaces
    text = re.sub(r"\s+", " ", text).strip()         # collapse spaces
    return text


def tokenize(text: str, remove_stopwords: bool = True,
             lemmatize: bool = True, min_len: int = 2) -> list:
    """Tokenize cleaned text; optionally drop stopwords and lemmatize."""
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return [t for t in tokens if len(t) >= min_len]


def preprocess_to_tokens(documents):
    """List of raw strings → list of token lists (for Word2Vec)."""
    return [tokenize(clean_text(doc)) for doc in documents]


def preprocess_to_strings(documents):
    """List of raw strings → list of cleaned space-joined strings (for TF-IDF)."""
    return [" ".join(tokenize(clean_text(doc))) for doc in documents]


if __name__ == "__main__":
    sample = "Hello! Email me at test@example.com or visit https://x.com. The cars are red."
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
    print("Tokens  :", tokenize(clean_text(sample)))
