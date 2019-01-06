"""Dynamic Topic Modelling

This file exposes a class that wraps gensim's `DtmModel` to add utils for
exploring topics, and it can be run as a script to train and persist a DTM.
"""
import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers import DtmModel
from scipy.stats import linregress

from src import HOME_DIR
from src.utils.corpus import Corpus
from src.utils.wiki2vec import label_topic


class Dtm(DtmModel):
    
    @classmethod
    def load(cls, fname):
        model_path = os.path.join(HOME_DIR, 'models', fname, 'dtm.gensim')
        obj = super().load(model_path)
        obj.__class__ = Dtm # TODO: remove when retrained
        obj._assign_corpus()
        return obj

    def _assign_corpus(self):
        """Assign corpus object to the model"""
        self.original_corpus = Corpus()
        assert self.original_corpus.debates.shape[0] == self.gamma_.shape[0]
        self.topic_assignments = self.get_topics_for_documents()
        self.time_slice_labels = self.original_corpus.debates.year.unique()

    def term_distribution(self, term, topic):
        """Extracts the probability over each time slice of a term/topic
        pair."""
        word_index = self.id2word.token2id[term]
        topic_slice = np.exp(self.lambda_[topic])
        topic_slice = topic_slice / topic_slice.sum(axis=0)
        return topic_slice[word_index]
    
    def term_variance(self, topic):
        """Finds variance of probability over time for terms for a given topic.
        High variance terms are more likely to be interesting than low variance
        terms."""
        p = np.exp(self.lambda_[topic]) /\
            np.exp(self.lambda_[topic]).sum(axis=0)
        variances = np.var(p, axis=1)
        order = np.argsort(variances)[::-1]
        terms = np.array([term for term, _
                        in sorted(self.id2word.token2id.items(),
                                  key=lambda x: x[1])])[order]
        variances = variances[order]
        return list(zip(terms, variances))
    
    def term_slope(self, topic):
        """Finds slope of probability over time for terms for a given topic.
        This is useful for roughly identifying terms that are rising or
        declining in popularity over time."""
        p = np.exp(self.lambda_[topic]) /\
            np.exp(self.lambda_[topic]).sum(axis=0)
        slopes = np.apply_along_axis(
            lambda y: linregress(x=range(len(y)), y=y).slope, axis=1, arr=p)
        order = np.argsort(slopes)
        terms = np.array([term for term, _
                        in sorted(self.id2word.token2id.items(),
                                    key=lambda x: x[1])])[order]
        slopes = slopes[order]
        return list(zip(terms, slopes))

    def plot_terms(self, topic, terms, title=None, name=None, hide_y=True):
        """Creates a plot of term probabilities over time in a given topic."""
        fig, ax = plt.subplots()
        plt.style.use('fivethirtyeight')
        for term in terms:
            ax.plot(
                self.time_slice_labels, self.term_distribution(term, topic),
                label=term)
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if hide_y:
            ax.set_yticklabels([])
        ax.set_ylabel('Probability')
        if title:
            ax.set_title(title)
        if name:
            fig.savefig(
                name, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
        return fig, ax
    
    def top_term_table(self, topic, slices, topn=10):
        """Returns a dataframe with the top n terms in the topic for each of
        the given time slices."""
        data = {}
        for time_slice in slices:
            time = np.where(self.time_slice_labels == time_slice)[0][0]
            data[time_slice] = [
                term for p, term
                in self.show_topic(topic, time=time, topn=topn)
            ]
        return pd.DataFrame(data)
    
    def summary(self, slices, topn=10):
        """Prints a summary of all the topics"""
        for topic in range(self.num_topics):
            print('Topic %d' % topic)
            print(self.top_term_table(topic, slices, topn))
            print()

    def topic_summary(self, topic, n=20):
        """Prints the top N terms by variance, increasing slope, and decreasing
        slope."""
        print('Variance\n---------')
        for row in self.term_variance(topic)[:n]:
            print(row)
        slopes = self.term_slope(topic)
        print('\nSlope (positive)\n----------')
        for row in slopes[-n:][::-1]:
            print(row)
        print('\nSlope (negative)\n----------')
        for row in slopes[:n]:
            print(row)

    def label_topic(self, i, time_slice, n=10):
        """Assign label to a given topic for a given time slice"""
        time = np.where(self.time_slice_labels == time_slice)[0][0]
        top_terms = [term for _, term in self.show_topic(i, time, n)]
        spacy_docs = self.get_spacy_docs_for_topic(i, time)
        return label_topic(spacy_docs, top_terms)
    
    def get_topics_for_documents(self):
        """Assign each document to its most probable topic"""
        p = np.exp(self.gamma_) /\
            np.exp(self.gamma_).sum(axis=1).reshape((-1,1))
        return np.apply_along_axis(np.argmax, 1, p)
    
    def get_spacy_docs_for_topic(self, i, time_slice):
        """Get spacy docs for documents matching a given topic in a given time
        slice"""
        indices = self.original_corpus.debates[
            ((self.original_corpus.debates.year -
              self.original_corpus.debates.year.min()) == time_slice) &
            (self.topic_assignments == i)].index
        return [self.original_corpus.paragraphs[j].spacy_doc() for j in indices]

def train(args, output_dir):
    """Build the corpus, trains the DTM, and saves the model to the output
    dir."""
    corpus = Corpus()

    # Create the dictionary.
    dictionary = Dictionary(corpus.debates.bag_of_words)
    dictionary.filter_extremes(no_below=100)

    # Train and save dtm.
    time_slices = corpus.debates.groupby('year').size()
    dtm_corpus = corpus.debates.bag_of_words.apply(dictionary.doc2bow)
    model = Dtm(
        args.executable, corpus=dtm_corpus, id2word=dictionary,
        num_topics=args.num_topics,
        time_slices=time_slices.values, rng_seed=args.random_seed
    )
    model.save(os.path.join(output_dir, 'dtm.gensim'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output',
        help='The name of the directory to output the model to (must not ' +
             'already exist). This will become a subdirectory under ' +
             '`models/`.',
        type=str,
        default=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))
    parser.add_argument(
        '-n', '--num-topics',
        help='The number of topics.',
        type=int,
        default=15)
    parser.add_argument(
        '-e', '--executable',
        help='The path to the DTM executable.',
        type=str,
        default='/home/lukelefebure/dtm/dtm/dtm')
    parser.add_argument(
        '-r', '--random-seed',
        help='Random seed.',
        type=int,
        default=5278)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(HOME_DIR, 'models', args.output)
    os.mkdir(output_dir)
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        filename=os.path.join(output_dir, 'log'))
    train(args, output_dir)
