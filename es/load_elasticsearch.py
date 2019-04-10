import argparse
import base64
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch, NotFoundError

from src.utils.corpus import Corpus

INDEX = 'debates'
DOC_TYPE = 'doc'

def load_data(es, subset=True):
    """If subset is true, only load data from China, US, Russia, UK, and France
    and year>1975. This puts the total under 10k which is the limit of the free
    tier for managed ES on Bonsai."""
    corpus = Corpus()
    def _data_generator():
        for _, row in corpus.debates.iterrows():
            if subset and (row.year <= 1975 or row.country_name not in \
                    ('United States Of America', 'China', 'Russia', 'France',
                     'United Kingdom')):
                continue
            data_dict = {
                'id': row.paragraph_id,
                'content': row.text,
                'country_code': row.country,
                'country': row.country_name,
                'year': row.year
            }
            op_dict = {
                "index": {
                    "_index": INDEX,
                    "_type": DOC_TYPE,
                    "_id": data_dict['id']
                }
            }
            yield op_dict
            yield data_dict
    resp = es.bulk(index=INDEX, body=_data_generator())
    return resp


def prepare_index(es):
    """Delete index (if it already exists), and recreate."""
    es.indices.delete(index=INDEX, ignore=[400, 404])
    request_body = {
        'settings' : {
            'number_of_shards': 5,
            'number_of_replicas': 1
        },
        'mappings': {
            DOC_TYPE: {
                'properties': {
                    'content': {
                        'type': 'text'
                    },
                    'country_code': {
                        'type': 'text'
                    },
                    'country': {
                        'type': 'keyword'
                    },
                    'year': {
                        'type': 'keyword'
                    }
                }
            }
        }
    }
    print(f"Creating {INDEX} index...")
    es.indices.create(index=INDEX, body=request_body)


def load_elasticsearch(args):
    if args.es_host != 'localhost':
        # Assume bonsai connection
        bonsai = args.es_host
        auth = re.search('https\:\/\/(.*)\@', bonsai).group(1).split(':')
        host = bonsai.replace('https://%s:%s@' % (auth[0], auth[1]), '')

        # Connect to cluster over SSL using auth for best security:
        es_header = {
            'host': host,
            'port': 443,
            'use_ssl': True,
            'http_auth': (auth[0], auth[1])
        }
    else:
        es_header = {
            'host': args.es_host,
            'port': args.es_port
        }
    es = Elasticsearch([es_header])
    prepare_index(es)
    load_data(es)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--es-host', default='localhost')
    parser.add_argument('-p', '--es-port', default='9200')
    args = parser.parse_args()
    load_elasticsearch(args)
