import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

import config


# WORD2VEC

def train_word2vec(tokenized_docs, epochs: int = None,
                   vector_size: int = None) -> Word2Vec:
    """Train a Skip-Gram Word2Vec model on tokenized documents."""
    epochs = epochs or config.W2V_EPOCHS_TO_COMPARE[-1]
    vector_size = vector_size or config.W2V_EMBED_DIM

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=config.W2V_WINDOW,
        min_count=config.W2V_MIN_COUNT,
        workers=config.W2V_WORKERS,
        sg=1,                       
        epochs=epochs,
        seed=config.RANDOM_SEED,
    )
    print(f"  Word2Vec: vocab={len(model.wv)}, dim={vector_size}, epochs={epochs}")
    return model


def doc_to_vec(tokens, w2v_model) -> np.ndarray:
    """Average the Word2Vec vectors of all known words in a document."""
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.wv.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def docs_to_matrix(tokenized_docs, w2v_model) -> np.ndarray:
    """Convert list of token lists to a (num_docs, embed_dim) matrix."""
    return np.array(
        [doc_to_vec(doc, w2v_model) for doc in tokenized_docs],
        dtype=np.float32,
    )


# TF-IDF

def fit_tfidf(train_strings, max_features: int = None) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on training strings only."""
    max_features = max_features or config.TFIDF_MAX_FEATURES
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,    
        min_df=2,             
        max_df=0.95,          
        norm="l2",
    )
    vec.fit(train_strings)
    print(f"  TF-IDF: vocab={len(vec.vocabulary_)}, max_features={max_features}")
    return vec


def texts_to_tfidf(strings, vectorizer) -> np.ndarray:
    """Transform strings into a dense TF-IDF matrix (float32)."""
    return np.asarray(vectorizer.transform(strings).todense(), dtype=np.float32)
