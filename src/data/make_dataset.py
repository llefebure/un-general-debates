import os
import pandas as pd
from src import HOME_DIR
from src.utils.tokenization import ParagraphTokenizer

def preprocess_data():
    """Load and preprocess the raw data

    This helper function loads and preprocesses the data to abstract away some
    of the menial work.

    Parameters
    ----------
    paragraph_tokenize : bool
        Indicate whether to paragraph tokenize the speeches.
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

    paragraph_tokenizer = ParagraphTokenizer()
    paragraphs = pd.Series(
        debates.text
        .apply(lambda x: [x[start:end] for start, end
                          in paragraph_tokenizer.span_tokenize(x)])
        .apply(lambda x: pd.Series(x))
        .stack()
        .reset_index(level=1, drop=True), name='text')
    debates_paragraphs = (debates
                          .drop('text', axis=1)
                          .join(paragraphs)
                          .reset_index())
    # Must retain this new index to preserve ordering of paragraphs within
    # each speech.
    debates_paragraphs.index.name = 'paragraph_index'

    # Save data to interim directory.
    debates.to_csv(
        os.path.join(HOME_DIR, 'data/processed/debates.csv'),
        index=False)
    debates_paragraphs.to_csv(
        os.path.join(HOME_DIR, 'data/processed/debates_paragraphs.csv'),
        index=True)

if __name__ == '__main__':
    preprocess_data()
