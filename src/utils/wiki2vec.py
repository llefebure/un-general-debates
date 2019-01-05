"""Utils for accessing and looking up pretrained Wikipedia2Vec [1] vectors

Wikipedia2vec provides pretrained word and entity (Wikipedia page title)
vectors. These serve several useful purposes including automatic discovery of
Wikipedia entities in the speeches and similarity between discovered entities.

.. [1] https://wikipedia2vec.github.io/wikipedia2vec/
"""
import logging
import numpy as np
import os
from collections import defaultdict, Counter
from wikipedia2vec import Wikipedia2Vec
from scipy.spatial.distance import cosine
from src import HOME_DIR

logger = logging.getLogger(__name__)

def _load_wikipedia2vec(
        wiki_model_path='data/external/enwiki_20180420_100d.pkl'):
    path = os.path.join(HOME_DIR, wiki_model_path)
    if os.path.exists(path):
        return Wikipedia2Vec.load(path)
    else:
        logger.warn('No pretrained Wikipedia2Vec found.')
        return None

wiki2vec = _load_wikipedia2vec()

def trim_pos(noun_chunk, pos='DET'):
    """Trims leading tokens of a given POS from a Span

    Parameters
    ----------
    noun_chunk : spacy.tokens.Span

    Returns
    -------
    spacy.tokens.Span
        Adjusted span with leading tokens of the given POS stripped.
    """
    start = 0
    for tok in noun_chunk:
        if tok.pos_ == pos:
            start += 1
        else:
            break
    return noun_chunk[start:]

def _casing_permutations(noun_chunk):
    """Generates casing permutations before wiki2vec entity lookup

    Case matters during lookup into the pretrained entity embeddings, so
    generate some permuatations to avoid misses.

    Parameters
    ----------
    noun_chunk : spacy.tokens.Span

    Returns
    -------
    list of str
        Different permutations of casing of the input.
    """
    return [noun_chunk.text.capitalize(), noun_chunk.text.title()]


def _permutations(noun_chunk):
    """Generate permutations of noun chunk before wiki2vec entity lookup

    Generates permutations of a noun chunk by stripping determiners (e.g."The")
    and other words one by one. The first of these permuatations to match a
    Wiki entity will be assigned as the entity for the noun chunk.

    Parameters
    ----------
    noun_chunk : spacy.tokens.Span

    Returns
    -------
    list of str
        Different variations on the noun chunk text to test for entity lookup.
    """
    permutations = []
    no_determiners = trim_pos(noun_chunk)
    for i in range(len(no_determiners)):
        trimmed = no_determiners[i:]
        if len(trimmed) > 1 or trimmed[0].pos_ != 'PRON':
            permutations.extend(_casing_permutations(trimmed))
    return permutations

def lookup_entity(noun_chunk):
    """Looks up entity for Span

    Parameters
    ----------
    noun_chunk : spacy.tokens.Span

    Returns
    -------
    wikipedia2vec.dictionary.Entity or None
        Entity matching the input or None if no matching entity found.
    """
    if not wiki2vec:
        logger.warn('Pretrained wikipedia2vec not loaded')
        return None
    for text in _permutations(noun_chunk):
        entity = wiki2vec.get_entity(text)
        if entity is not None:
            return entity
    return None

def label_topic(spacy_docs, top_terms, n=10):
    """Assign entity labels to a topic

    Parameters
    ----------
    spacy_docs : list of spacy.tokens.Doc
        Representative docs from a topic learned by a topic model.
    top_terms : list of str
        List of representative terms from a topic.
    n : int
        Number of entity labels to return.

    Returns
    -------
    list of tuples
        List of entity labels along with a count of the entity in the provided
        list of documents and a score measuring relevance of the entity to the
        provided term list.
    """
    entities = Counter()
    for doc in spacy_docs:
        for nc in doc.noun_chunks:
            if nc._.entity:
                entities[nc._.entity.title] += 1
    final_candidates = list()
    for candidate, count in entities.most_common(n):
        scores = np.array([
            1 - cosine(
                wiki2vec.get_entity_vector(candidate),
                wiki2vec.get_word_vector(term))
            for term in top_terms if wiki2vec.get_word(term)
        ])
        final_candidates.append((candidate, count, scores.mean()))
    return sorted(final_candidates, key=lambda x: -x[2])
