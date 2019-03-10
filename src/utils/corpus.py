"""Utils for loading and adding annotations to the data"""
import ast
import logging
import msgpack
import os
import pandas as pd
from functools import lru_cache
from spacy.tokens import Doc
from tqdm import tqdm

from src import HOME_DIR
from src.utils.spacy import nlp, all_pipes

logger = logging.getLogger(__name__)
cache = lru_cache(maxsize=None)


def _load_spacy():
    """Loads serialized spacy vocab and docs

    Returns
    -------
    dict
        Maps doc index to bytes for spacy doc.
    """
    spacy_path = os.path.join(HOME_DIR, 'data/processed/spacy')
    if os.path.exists(spacy_path):
        with open(spacy_path, 'rb') as f:
            m = msgpack.load(f)
        nlp.vocab.from_bytes(m[b'vocab'])
        return {i: doc_bytes for i, doc_bytes in enumerate(m[b'docs'])}
    else:
        logger.warn('No serialized Spacy found')
        return None


class Paragraph:
    """A paragraph from the corpus

    Parameters
    ----------
    index : int
        Index of the paragraph within the speech.
    paragraph_id : int
        A globally unique id for the paragraph.
    row : pd.Series
        The row of data referring to this speech.
    parent : src.corpus.Speech
        A reference to the Speech object that contains this paragraph.
    """
    def __init__(self, index, paragraph_id, row, parent):
        self.index = index
        self.id_ = paragraph_id
        self.row = row
        self.speech = parent

    def series(self):
        return self.row
    
    def spacy_doc(self):
        return self.speech.spacy_paragraphs()[self.index]
    
    def session(self):
        return self.row.session
    
    def year(self):
        return self.row.year

    def country_code(self):
        return self.row.country
    
    def country(self):
        return self.row.country_name

class Speech:
    """A speech from the corpus

    Serialized Spacy docs are lazy loaded.

    Parameters
    ----------
    speech_id : int
        A globally unique id for the speech.
    group : pd.DataFrame
        The subset of rows/paragraphs that belong to this speech.
    spacy_bytes : bytes
        Serialized spacy doc.
    """
    def __init__(self, speech_id, group, spacy_bytes=None):
        self.id_ = speech_id
        self._spacy_bytes = spacy_bytes
        self.group = group
        self.paragraphs = [
            Paragraph(i, id_, row, self)
            for i, (id_, row) in enumerate(group.iterrows())
        ]

    @cache
    def spacy_doc(self):
        if self._spacy_bytes is not None:
            doc = Doc(nlp.vocab).from_bytes(self._spacy_bytes)
            for f in all_pipes:
                doc = f(doc)
            return doc
        else:
            raise FileNotFoundError('No serialized Spacy found')

    def spacy_paragraphs(self):
        return self.spacy_doc()._.paragraphs

    def session(self):
        return self.group.session.iloc[0]

    def year(self):
        return self.group.year.iloc[0]

    def country_code(self):
        return self.group.country.iloc[0]

    def country(self):
        return self.group.country_name.iloc[0]

class Corpus:
    """UN General Debate Corpus"""
    def __init__(self, filename='data/processed/debates_paragraphs.csv',
                 from_year=None):
        self.filename = filename
        self._load(filename)

    def _load(self, filename):
        debates = pd.read_csv(os.path.join(HOME_DIR, filename))
        debates.bag_of_words = debates.bag_of_words.apply(ast.literal_eval)
        spacy = _load_spacy()
        self.debates = debates
        self.speeches = [
            Speech(i, group, spacy.pop(i) if spacy else None)
            for i, group in debates.groupby('index')
        ]
        self.paragraphs = [par for sp in self.speeches for par in sp.paragraphs]

    def add_dataframe_column(self, column):
        """Add column to the dataframe

        Add a column to the corpus dataframe and save it so that it loads next
        time. Useful for adding paragraph level annotations that don't
        necessarily make sense as a Spacy extension.

        Parameters
        ----------
        column : pd.Series
            New column to append to the corpus dataframe. Should be named.
        """
        self.debates = pd.concat([self.debates, column], axis=1)
        self.debates.to_csv(self.filename, index=False)

    def load_spacy_cache(self):
        for sp in tqdm(self.speeches):
            sp.spacy_doc()
