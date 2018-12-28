import os
import pandas as pd
import msgpack
from tqdm import tqdm
from src import HOME_DIR
from src.utils.spacy import nlp, paragraph_tokenizer, bow

def preprocess_data():
    """Preprocess the raw data

    This helper function preprocesses the data by merging together country codes
    with their full names, serializing Spacy markup of every speech, and saving
    a CSV output. See `src/utils/corpus.py` for utils for loading.
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
    debates = debates.sort_values(['year', 'country']).reset_index(drop=True)
    debates.to_csv(
        os.path.join(HOME_DIR, 'data/processed/debates.csv'),
        index=False)

    # Compute and serialize Spacy
    if input('Compute Spacy [yn]:') == 'y':
        output = {'docs': []}
        docs = []
        for doc in tqdm(
                nlp.pipe((t.strip() for t in debates.text), batch_size=20),
                total=debates.shape[0]):
            docs.append(doc)
            output['docs'].append(doc.to_bytes(tensor=False))
        output['vocab'] = nlp.vocab.to_bytes()
        filename = os.path.join(HOME_DIR, 'data/processed/spacy')
        with open(filename, 'wb') as f:
            f.write(msgpack.dumps(output))
        paragraphs = pd.Series(
            pd.Series(debates.index)
            .apply(lambda x: bow(paragraph_tokenizer(docs[x]))._.paragraphs)
            .apply(lambda x: pd.Series(x))
            .stack()
            .reset_index(level=1, drop=True), name='spacy_paragraph')
        derived_paragraph_features = pd.DataFrame({
            'text': paragraphs.apply(lambda x: x.text),
            'bag_of_words': paragraphs.apply(lambda x: x._.bow)
        })
        debates_paragraphs = (
            debates
            .drop('text', axis=1)
            .join(derived_paragraph_features)
            .reset_index())
        debates_paragraphs.index.name = 'paragraph_index'
        debates_paragraphs.to_csv(
            os.path.join(HOME_DIR, 'data/processed/debates_paragraphs.csv'),
            index=False)

if __name__ == '__main__':
    preprocess_data()
