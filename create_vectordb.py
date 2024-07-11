#!/usr/bin/python3

from absl import flags, app
from elasticsearch import Elasticsearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = None, help = 'elastic search host')
  flags.DEFINE_string('index', default = 'qd_asset', help = 'index')
  flags.DEFINE_string('username', default = 'elastic', help = 'username')
  flags.DEFINE_string('password', default = None, help = 'password')

def main(unused_argv):
  host_with_authentication = FLAGS.host[:FLAGS.host.find('://') + 3] + FLAGS.username + ":" + FLAGS.password + "@" + FLAGS.host[FLAGS.host.find('://') + 3:]
  es = Elasticsearch(host_with_authentication)
  res = es.search(index = FLAGS.index,
                  scroll = "1m",
                  size = 100,
                  body = {
                    "query": {"match_all": {}}
                  })
  scroll_id = res['_scroll_id']
  hits = res['hits']['hits']

  embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Chroma(embedding = embeddings, persist_directory = 'db')
  while len(hits) > 0:
    texts = [hit['资产详细信息'] for hit in hits]
    metadatas = [{'_id': hit['_id'] for hit in hits}]
    vectordb.add_texts(texts = texts, metadatas = metadatas)
    res = es.scroll(scroll_id = scroll_id, scroll = "1m")
    scroll_id = res['_scroll_id']
    hits = res['hits']['hits']

  es.clear_scroll(scroll_id = scroll_id)

if __name__ == "__main__":
  add_options()
  app.run(main)

