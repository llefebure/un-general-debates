"""Utils for computing tfidf"""
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel

def generate_tfidf(corpus_df, dictionary):
    """Generates TFIDF matrix for the given corpus.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        The corpus dataframe.
    dictionary : gensim.corpora.dictionary.Dictionary
        Dictionary defining the vocabulary of the TFIDF.

    Returns
    -------
    X : np.ndarray
        TFIDF matrix with documents as rows and vocabulary as the columns.
    """
    tfidf_model = TfidfModel(
        corpus_df.bag_of_words.apply(lambda x: dictionary.doc2bow(x)))
    model = tfidf_model[
        corpus_df.bag_of_words.apply(lambda x: dictionary.doc2bow(x))]
    X = corpus2csc(model, len(dictionary)).T
    return X
