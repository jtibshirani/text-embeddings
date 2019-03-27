from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import json
import tensorflow as tf
import tensorflow_hub as hub
import time

def reindex_docs():
  print("Creating the 'posts' index.")
  client.indices.delete(index=INDEX_NAME, ignore=[404])

  with open(INDEX_FILE) as file:
    source = file.read().strip()
    client.indices.create(index=INDEX_NAME, body=source)

  index_documents()

def index_documents():
  print("Indexing documents...")

  bulk_requests = []
  titles = []
  index = 0

  with open(DATA_FILE) as file:
    while True:
      line = file.readline().strip()
      if not line:
        break

      source = json.loads(line)
      if source['type'] != 'question':
        continue

      source.update({'_op_type': 'index',
        '_index': INDEX_NAME,
        '_id': index})
      
      bulk_requests.append(source)
      titles.append(source['title'])
      index += 1

      if index % BATCH_SIZE == 0:
        index_batch(index, bulk_requests, titles)
        bulk_requests = []
        titles = []

    if bulk_requests:
      index_batch(index, bulk_requests, titles)

  client.indices.refresh(index=INDEX_NAME)
  print('Done indexing.')

def index_batch(index, bulk_requests, titles):
  print("Calculating embeddings for batch...")
  title_vectors = embed_text(titles)

  for i, request in enumerate(bulk_requests):
    request['title_vector'] = title_vectors[i]

  print("Indexing documents...")
  bulk(client, bulk_requests)

  print("Indexed {} documents.".format(index))

def embed_text(text):
    vectors = session.run(tensors, feed_dict={text_ph: text})

    # The dense_vector field allows a maximum of 499 dimensions, whereas
    # universal-sentence-encoder produces vectors with 512 dimensions.
    return [vector.tolist()[:499] for vector in vectors]


def start_query_loop():
  while True:
    query = input('Enter query: ')

    if query == 'quit':
      break

    encoding_start = time.time()

    query_vector = embed_text([query])[0]
    encoding_time = time.time() - encoding_start

    script_query = {
      "script_score": {
        "query": {
          "match_all": {}
        },
        "script": {
          "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
          "params": {
            "query_vector": query_vector
          }
        }
      }
    }

    search_start = time.time()
    response = client.search(index=INDEX_NAME, body={'size': 5,
      'query': script_query,
      '_source': { 'includes': ['title', 'body']}})
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response['hits']['total']['value']))
    print("Encoding time: {:.2f} ms".format(encoding_time * 1000), "search time: {:.2f} ms".format(search_time * 1000))
    for hit in response['hits']['hits']:
      print('id: {}, score: {}'.format(hit['_id'], hit['_score']))
      print(hit['_source'])
      print()

##### MAIN SCRIPT #####

INDEX_NAME = 'posts'
INDEX_FILE = 'data/posts/index.json'

DATA_FILE = 'data/posts/posts_100.json'
BATCH_SIZE = 100

print("Downloading pre-trained embeddings from tensorflow hub.")
use_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
#elmo_encoder = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

text_ph = tf.placeholder(tf.string)
tensors = use_encoder(text_ph)

print("Creating a tensorflow session.")
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())

client = Elasticsearch()

reindex_docs()
start_query_loop()

session.close()

