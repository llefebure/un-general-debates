"""Utils for using Spacy"""
import spacy
from spacy.tokens import Doc, Span
from src.utils.wiki2vec import lookup_entity


class ParagraphTokenizer:
    """Spacy extension for paragraph tokenization"""
    def __init__(self):
        Doc.set_extension('paragraphs', default=[])
    
    def __call__(self, doc):
        spans = []
        start = 0
        for sent in doc.sents:
            if '\n' in doc[sent.end-1].text_with_ws:
                spans.append(Span(doc, start, sent.end))
                start = sent.end
        if start != len(doc):
            spans.append(Span(doc, start, len(doc)))
        doc._.set('paragraphs', spans)
        return doc

class BagOfWords:
    """Spacy extension for appending bag of words feature"""
    def __init__(self):
        Doc.set_extension('bow', default=[])
        Span.set_extension('bow', default=[])

    def __call__(self, doc):
        doc_bow = []
        for par in doc._.paragraphs:
            par_bow = [
                tok.lemma_.lower() for tok in par
                if not (tok.is_space or tok.is_punct or tok.is_stop or
                       (tok.is_sent_start and tok.is_digit))
            ]
            par._.set('bow', par_bow)
            doc_bow.extend(par_bow)
        doc._.set('bow', doc_bow)
        return doc

class Entity:
    """Spacy extension for appending wiki entities"""
    def __init__(self):
        Span.set_extension('entity', default=None)

    def __call__(self, doc):
        for nc in doc.noun_chunks:
            nc._.set('entity', lookup_entity(nc))
        return doc


# This is pretty ugly and confusing. `Doc.to_bytes` fails when these extensions
# are registered because it doesn't know how to serialize the types. Therefore,
# I just apply the extensions after when I need them with `apply_extensions`.
# In an ideal world, we'd be using `nlp.add_pipe`.
nlp = spacy.load('en')

extensions = [
    ParagraphTokenizer(),
    BagOfWords(),
    Entity()
]

def apply_extensions(x):
    for ext in extensions:
        x = ext(x)
    return x
