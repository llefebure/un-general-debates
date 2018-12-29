"""Utils for loading and exploring trained DTMs."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from gensim.models.wrappers import DtmModel
from scipy.stats import linregress

from src import HOME_DIR

def load_dtm(name):
    """Helper function to load in the model."""
    model_path = os.path.join(HOME_DIR, 'models', name, 'dtm.gensim')
    time_slices = pd.read_pickle(
        os.path.join(HOME_DIR, 'models', name, 'time_slices.p'))
    return DtmModel.load(model_path), time_slices

def term_distribution(model, term, topic):
    """Extracts the probability over each time slice of a term/topic pair."""
    word_index = model.id2word.token2id[term]
    topic_slice = np.exp(model.lambda_[topic])
    topic_slice = topic_slice / topic_slice.sum(axis=0)
    return topic_slice[word_index]

def term_variance(model, topic):
    """Finds variance of probability over time for terms for a given topic.
    High variance terms are more likely to be interesting than low variance
    terms."""
    p = np.exp(model.lambda_[topic]) /\
        np.exp(model.lambda_[topic]).sum(axis=0)
    variances = np.var(p, axis=1)
    order = np.argsort(variances)[::-1]
    terms = np.array([term for term, _
                      in sorted(model.id2word.token2id.items(),
                                key=lambda x: x[1])])[order]
    variances = variances[order]
    return list(zip(terms, variances))

def term_slope(model, topic):
    """Finds slope of probability over time for terms for a given topic. This
    is useful for roughly identifying terms that are rising or declining in
    popularity over time."""
    p = np.exp(model.lambda_[topic]) /\
        np.exp(model.lambda_[topic]).sum(axis=0)
    slopes = np.apply_along_axis(
        lambda y: linregress(x=range(len(y)), y=y).slope, axis=1, arr=p)
    order = np.argsort(slopes)
    terms = np.array([term for term, _
                      in sorted(model.id2word.token2id.items(),
                                key=lambda x: x[1])])[order]
    slopes = slopes[order]
    return list(zip(terms, slopes))

def plot_terms(x, model, topic, terms, title=None, name=None, hide_y=True):
    """Creates a plot of term probabilities over time in a given topic."""
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')

    for term in terms:
        ax.plot(x, term_distribution(model, term, topic), label=term)
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

def top_term_table(model, topic, slices, slice_labels, topn=10):
    """Returns a dataframe with the top n terms in the topic for each of
    the given time slices."""
    data = {}
    for time_slice in slices:
        data[slice_labels[time_slice]] = [
            term for p, term 
            in model.show_topic(topic, time=time_slice, topn=topn)
        ]
    return pd.DataFrame(data)

def summary(model, topic, n=20):
    """Prints the top N terms by variance, increasing slope, and decreasing slope."""
    print('Variance\n---------')
    for row in term_variance(model, topic)[:n]:
        print(row)
    slopes = term_slope(model, topic)
    print('\nSlope (positive)\n----------')
    for row in slopes[-n:][::-1]:
        print(row)
    print('\nSlope (negative)\n----------')
    for row in slopes[:n]:
        print(row)
