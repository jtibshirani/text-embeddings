from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import json
import time

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

##### INDEXING #####

def ingest_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as file:
        source = file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)
    index_docs()

def index_docs():
    bulk_requests = []
    titles = []
    id = 0

    with open(DATA_FILE) as file:
        while True:
            line = file.readline().strip()
            if not line: break

            source = json.loads(line)
            if source["type"] != "question":
                continue

            source.update({"_op_type": "index", "_index": INDEX_NAME, "_id": id})

            bulk_requests.append(source)
            titles.append(source["title"])
            id += 1

            if id % BATCH_SIZE == 0:
                index_batch(bulk_requests, titles)
                bulk_requests = []
                titles = []
                print("Indexed {} documents.".format(id))

        if bulk_requests:
            index_batch(bulk_requests, titles)
            print("Indexed {} documents.".format(id))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")

def index_batch(bulk_requests, titles):
    title_vectors = embed_text(titles)
    for i, request in enumerate(bulk_requests):
        request["title_vector"] = title_vectors[i]
    bulk(client, bulk_requests)

##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "body"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()

##### EMBEDDING #####

def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]

##### MAIN SCRIPT #####

INDEX_NAME = "posts"
INDEX_FILE = "data/posts/index.json"

DATA_FILE = "data/posts/posts.json"
BATCH_SIZE = 1000

SEARCH_SIZE = 5

print("Downloading pre-trained embeddings from tensorflow hub...")
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
text_ph = tf.placeholder(tf.string)
embeddings = embed(text_ph)
print("Done.")

print("Creating tensorflow session...")
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())
print("Done.")

client = Elasticsearch()

ingest_data()
run_query_loop()

print("Closing tensorflow session...")
session.close()
print("Done.")
