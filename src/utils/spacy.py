import spacy
from spacy.tokens import Doc, Span

class ParagraphTokenizer:
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

nlp = spacy.load('en')
paragraph_tokenizer = ParagraphTokenizer()
