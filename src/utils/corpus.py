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
from src.utils.spacy import nlp, paragraph_tokenizer

def generate_tfidf(corpus, dictionary):
    """Generates TFIDF matrix for the given corpus.

    Params
    ------
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
        corpus.bag_of_words.apply(lambda x: dictionary.doc2bow(x)))
    model = tfidf_model[
        corpus.bag_of_words.apply(lambda x: dictionary.doc2bow(x))]
    X = corpus2csc(model, len(dictionary)).T
    return X

def load_corpus(split_paragraphs=True):
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

    # deserialize spacy
    m = msgpack.load(
        open(os.path.join(HOME_DIR, 'data/processed/spacy'), 'rb'))
    nlp.vocab.from_bytes(m[b'vocab'])
    docs = {}
    for doc_id in m[b'docs']:
        doc = paragraph_tokenizer(
            Doc(nlp.vocab).from_bytes(m[b'docs'][doc_id]))
        docs[doc_id] = doc

    debates = pd.read_csv(os.path.join(HOME_DIR, 'data/processed/debates.csv'))

    if split_paragraphs:
        paragraphs = pd.Series(
            pd.Series(debates.index)
            .apply(lambda x: docs[x]._.paragraphs)
            .apply(lambda x: pd.Series(x))
            .stack()
            .reset_index(level=1, drop=True), name='text')
        debates = (debates
                    .drop('text', axis=1)
                    .join(paragraphs)
                    .reset_index())
        debates.index.name = 'paragraph_index'
    else:
        debates.text = pd.Series(debates.index).apply(lambda x: docs[x])

    return debates, docs
