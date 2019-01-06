"""DTM Training

This script trains a DTM and saves the resulting model.
"""
import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers import DtmModel

from src import HOME_DIR
from src.utils.corpus import Corpus


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
    model = DtmModel(
        args.executable, corpus=dtm_corpus, id2word=dictionary,
        num_topics=args.num_topics,
        time_slices=time_slices.values, rng_seed=args.random_seed
    )
    time_slices.to_pickle(os.path.join(output_dir, 'time_slices.p'))
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
