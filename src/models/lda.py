"""LDA with labelled topics"""    
from functools import lru_cache
from gensim.models import LdaModel
from src.utils.wiki2vec import label_topic


class Lda(LdaModel):
    """Wrapper for gensim lda model to allow for topic labelling

    Parameters
    ----------
    original_corpus : src.utils.corpus.Corpus
    dictionary : gensim.corpora.dictionary.Dictionary
    **kwargs
        Forwarded to gensim.models.LdaMulticore
    """
    def __init__(self, original_corpus, dictionary, **kwargs):
        self.original_corpus = original_corpus
        self.dictionary = dictionary
        super().__init__(**kwargs)
        self.topic_assignments = self.get_topics_for_documents()

    def label_topic(self, i, n=10):
        """Assign label to a given topic"""
        top_terms = [term for term, _ in self.show_topic(i, n)]
        spacy_docs = self.get_spacy_docs_for_topic(i)
        return label_topic(spacy_docs, top_terms)

    def get_topics_for_documents(self):
        """Assign each document to its most probable topic"""
        bow = self.original_corpus.debates.bag_of_words.apply(
            self.dictionary.doc2bow)
        def _get_topic(x):
            topics = self.get_document_topics(x)
            s = sorted(topics, key=lambda x: -x[1])
            return s[0][0]
        return bow.apply(lambda x: _get_topic(x))

    def get_spacy_docs_for_topic(self, i):
        """Get spacy docs for documents matching a given topic"""
        docs = [
            self.original_corpus.paragraphs[j].spacy_doc()
            for j in self.topic_assignments[self.topic_assignments == i].index
        ]
        return docs
