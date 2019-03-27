# Text Embeddings Prototype

The simple script `src/main.py` indexes the first ~20,000 questions from the
[StackOverflow](https://github.com/elastic/rally-tracks/tree/master/so)
dataset. Before indexing, each post's title is run through a pre-trained sentence embedding to
produce a [`dense_vector`](https://www.elastic.co/guide/en/elasticsearch/reference/master/dense-vector.html).

The script then accepts free-text queries in a loop ("Enter query: ..."). We first run the text through
the same sentence embedding to produce a vector, then perform a search based on
[cosine similarity](https://www.elastic.co/guide/en/elasticsearch/reference/7.x/query-dsl-script-score-query.html#vector-functions).

Currrently Google's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) is used
to perform the vector embedding. This is a fully pre-trained model, and no 'fine tuning' is performed
on the StackOverflow dataset.

# Usage

`pip3 install elasticsearch tensorflow tensorflow_hub`

`python3 src/main.py`

On subsequent runs, comment out `reindex_docs()` in the script to avoid repopulating the index.

# Example Queries

The following queries return good posts in the top position, despite there not being strong term
overlap between the query and document title:

- "zipping up files" returns "Compressing / Decompressing Folders & Files"
- "find location of an IP" returns "How to get the Country according to a certain IP?"

# Known Issues

- Universal Sentence Encoder produces vectors of length 512, but `dense_vector` dimension is capped at 500
- need to add 1 to `cosineSimilarity` to avoid negative scores
