"""Utils for accessing and looking up pretrained Wikipedia2Vec [1] vectors

Wikipedia2vec provides pretrained word and entity (Wikipedia page title)
vectors. These serve several useful purposes including automatic discovery of
Wikipedia entities in the speeches and similarity between discovered entities.

.. [1] https://wikipedia2vec.github.io/wikipedia2vec/
"""
import logging
import os
from collections import defaultdict
from wikipedia2vec import Wikipedia2Vec
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
