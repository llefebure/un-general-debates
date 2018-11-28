import os
import pandas as pd
from src import HOME_DIR

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
