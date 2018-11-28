"""This script trains the DTM and saves the resulting model.

To see the command line options, run:
    $ python src/models/train_dtm.py --help
"""
import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models.wrappers import DtmModel

from src import HOME_DIR
from src.utils.tokenization import (ParagraphTokenizer, WordTokenizer,
    SentenceTokenizer)
from src.utils.corpus import load_corpus

def train(args, output_dir):
    """Build the corpus, trains the DTM, and saves the model to the output
    dir."""
    debates = load_corpus()

    # Generate BOW representation of the docs.
    word_tokenizer = WordTokenizer()
    debates['bag_of_words'] = (
        debates.text.apply(lambda x: word_tokenizer.tokenize(x)))

    # Create the dictionary.
    dictionary = Dictionary(debates.bag_of_words)
    dictionary.filter_extremes(no_below=100)

    # Train and save dtm.
    corpus = debates.bag_of_words.apply(lambda x: dictionary.doc2bow(x))
    model = DtmModel(
        args.executable, corpus=corpus, id2word=dictionary,
        num_topics=args.num_topics,
        time_slices=debates.groupby('year').size().values, rng_seed=5278
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
