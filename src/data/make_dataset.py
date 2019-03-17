import os
import msgpack
import pandas as pd
import string
from tqdm import tqdm
from src import HOME_DIR
from src.utils.spacy import nlp, apply_extensions


def preprocess_data():
    """Preprocess the raw data

    This helper function preprocesses the data by merging together country codes
    with their full names, serializing Spacy markup of every speech, and saving
    two CSV outputs, one before paragraph tokenization and one after. See
    `src/utils/corpus.py` for utils for loading.
    """
    debates = pd.read_csv(
        os.path.join(HOME_DIR, 'data/external/un-general-debates.csv'))

    iso_codes = pd.read_csv(
        os.path.join(HOME_DIR,
                     'data/external/wikipedia-iso-country-codes.csv'),
        usecols=['English short name lower case', 'Alpha-3 code'])
    iso_codes.columns = ['country_name', 'country']

    # Certain codes were missing, so need to add manually.
    iso_codes = iso_codes.append(
        pd.DataFrame({
            'country_name': ['Democratic Yemen', 'Czechoslovakia',
                             'Yugoslavia', 'East Germany', 'European Union',
                             'South Sudan'],
            'country': ['YDYE', 'CSK', 'YUG', 'DDR', 'EU', 'SSD']
        }),
        sort=False
    )
    debates = pd.merge(debates, iso_codes, how='left', on='country')
    debates.text = debates.text.apply(lambda x: x.replace('\ufeff', '').strip())
    debates.sort_values(['year', 'country'], inplace=True)
    debates.reset_index(drop=True, inplace=True)
    debates.index.name = 'document_id'
    debates.to_csv(
        os.path.join(HOME_DIR, 'data/processed/debates.csv'),
        index=True)

    # Compute and serialize Spacy.
    output = {'docs': {}}
    docs = {}
    for doc_id, doc in tqdm(zip(
            debates.index,
            nlp.pipe((t for t in debates.text), batch_size=20)),
            total=debates.shape[0]):
        docs[doc_id] = doc
        output['docs'][doc_id] = doc.to_bytes(tensor=False)
    output['vocab'] = nlp.vocab.to_bytes()
    filename = os.path.join(HOME_DIR, 'data/processed/spacy')
    with open(filename, 'wb') as f:
        f.write(msgpack.dumps(output))

    # Paragraph tokenize and save second csv.
    paragraphs = (
        pd.Series(debates.index)
        .apply(lambda x: apply_extensions(docs[x])._.paragraphs)
        .apply(lambda x: pd.Series(x)))
    paragraphs.index = debates.index
    paragraphs = paragraphs.stack()
    paragraph_ids = ['_'.join(map(str, x)) for x in zip(
        paragraphs.index.labels[0], paragraphs.index.labels[1])]
    derived_paragraph_features = pd.DataFrame({
        'text': paragraphs.apply(lambda x: x.text),
        'bag_of_words': paragraphs.apply(lambda x: x._.bow),
        'paragraph_id': paragraph_ids
    })
    debates_paragraphs = (
        debates
        .drop('text', axis=1)
        .join(derived_paragraph_features))
    debates_paragraphs.index.rename(
        ['document_id', 'paragraph_index'], inplace=True)
    debates_paragraphs.reset_index(inplace=True)
    debates_paragraphs.sort_values(
        ['year', 'country', 'paragraph_index'], inplace=True)
    debates_paragraphs.to_csv(
        os.path.join(HOME_DIR, 'data/processed/debates_paragraphs.csv'),
        index=False)

if __name__ == '__main__':
    preprocess_data()
