"""Utils for loading the data"""
import ast
import glob
import msgpack
import os
import pandas as pd
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel
from spacy.tokens import Doc, Span
from src import HOME_DIR
from src.utils.spacy import nlp, paragraph_tokenizer, bow

SPACY_DOCS = None

def _deserialize_spacy():
    """Deserializes spacy vocab and docs

    Returns
    -------
    docs : dict
        Maps doc index to `spacy.tokens.Doc`.
    """
    global SPACY_DOCS
    if SPACY_DOCS is not None:
        return SPACY_DOCS
    SPACY_DOCS = []
    m = msgpack.load(
        open(os.path.join(HOME_DIR, 'data/processed/spacy'), 'rb'))
    nlp.vocab.from_bytes(m[b'vocab'])
    for doc_bytes in m[b'docs']:
        doc = bow(paragraph_tokenizer(
            Doc(nlp.vocab).from_bytes(doc_bytes)))
        SPACY_DOCS.append(doc)
    return SPACY_DOCS


def generate_tfidf(corpus, dictionary):
    """Generates TFIDF matrix for the given corpus.

    Parameters
    ----------
    corpus : pd.DataFrame
        The corpus loaded from `load_corpus`.
    dictionary : gensim.corpora.dictionary.Dictionary
        Dictionary defining the vocabulary of the TFIDF.

    Returns
    -------
    X : np.ndarray
        TFIDF matrix with documents as rows and vocabulary as the columns.
    """
    tfidf_model = TfidfModel(
        corpus.text.apply(lambda x: dictionary.doc2bow(x.bag_of_words)))
    model = tfidf_model[
        corpus.text.apply(lambda x: dictionary.doc2bow(x.bag_of_words))]
    X = corpus2csc(model, len(dictionary)).T
    return X

def load_corpus(split_paragraphs=True, load_spacy=False):
    """Loads preprocessed data

    Parameters
    ----------
    split_paragraphs : bool
        Indicate whether to split speeches into paragraphs.

    Returns
    -------
    debates : pd.DataFrame
    docs : dict
        Maps document index to `spacy.tokens.Doc`
    """
    if split_paragraphs:
        filename = 'data/processed/debates_paragraphs.csv'
    else:
        filename = 'data/processed/debates.csv'
    debates = pd.read_csv(os.path.join(HOME_DIR, filename))
    if 'bag_of_words' in debates.columns:
        debates.bag_of_words = debates.bag_of_words.apply(ast.literal_eval)
    if load_spacy:
        docs = _deserialize_spacy()
        if split_paragraphs:
            debates['spacy_text'] = [
                par for doc in docs for par in doc._.paragraphs
            ]
        else:
            debates['spacy_text'] = docs
    return debates
