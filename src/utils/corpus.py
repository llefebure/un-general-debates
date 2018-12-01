"""Utils for loading the data"""
import os
import pandas as pd
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from src import HOME_DIR
from src.utils.tokenization import WordTokenizer

def generate_tfidf(corpus):
    """Generates TFIDF matrix for the given corpus.

    Params
    ------
    corpus : pd.DataFrame
        The corpus loaded from `load_corpus`.

    Returns
    -------
    X : np.ndarray
        TFIDF matrix with documents as rows and vocabulary as the columns.
    dictionary : gensim.corpora.dictionary.Dictionary
        Dictionary defining the vocabulary of the TFIDF.
    """
    word_tokenizer = WordTokenizer()
    corpus['bag_of_words'] = (
        corpus.text.apply(lambda x: word_tokenizer.tokenize(x)))
    dictionary = Dictionary(corpus.bag_of_words)
    dictionary.filter_extremes(no_below=100)
    tfidf_model = TfidfModel(
        corpus.bag_of_words.apply(lambda x: dictionary.doc2bow(x)))
    model = tfidf_model[
        corpus.bag_of_words.apply(lambda x: dictionary.doc2bow(x))]
    X = corpus2dense(model, len(dictionary)).T
    return X, dictionary

def load_corpus(paragraph_tokenize=True):
    """Loads preprocessed data

    Parameters
    ----------
    paragraph_tokenize : bool
        Indicate whether to paragraph tokenize the speeches.
    """
    if paragraph_tokenize:
        fname = 'data/processed/debates_paragraphs.csv'
    else:
        fname = 'data/processed/debates.csv'
    return pd.read_csv(os.path.join(HOME_DIR, fname))
