"""Utils for loading the data"""
import ast
import os
import pandas as pd
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel
from src import HOME_DIR
from src.utils.tokenization import WordTokenizer

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

def load_corpus(paragraph_tokenize=True):
    """Loads preprocessed data

    Parameters
    ----------
    paragraph_tokenize : bool
        Indicate whether to paragraph tokenize the speeches.

    Returns
    -------
    df : pd.DataFrame
    """
    if paragraph_tokenize:
        fname = 'data/processed/debates_paragraphs.csv'
    else:
        fname = 'data/processed/debates.csv'
    df = pd.read_csv(os.path.join(HOME_DIR, fname))
    if 'bag_of_words' in df:
        df.bag_of_words = df.bag_of_words.apply(ast.literal_eval)
    return df
