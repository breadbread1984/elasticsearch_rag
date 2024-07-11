#!/usr/bin/python3

from absl import flags, app
from json_repair
from elasticsearch import Elasticsearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = None, help = 'elastic search host')
  flags.DEFINE_string('index', default = 'qd_asset', help = 'index')
  flags.DEFINE_string('username', default = 'elastic', help = 'username')
  flags.DEFINE_string('password', default = None, help = 'password')
  flags.DEFINE_integer('total', default = None, help = 'total records')

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
  vectordb = Chroma(embedding_function = embeddings, persist_directory = 'db')
  count = 0
  while len(hits) > 0:
    for hit in hits:
      texts = list()
      metadatas = list()
      _id = {'_id': hit['_id']}
      if '资产详细信息' in hit['_source']:
        detail = json_repair.loads(hit['_source']['资产详细信息'])
        for k, v in detail.items():
          texts.append('%s:%s' % (k,v))
          metadatas.append(_id)
      if '对应字段信息' in hit['_source']:
        domain = json_repair.loads(hit['_source']['对应字段信息'])
        for k, v in detail.items():
          texts.append('%s:%s' % (k,v))
          metadatas.append(_id)
    vectordb.add_texts(texts = texts, metadatas = metadatas)
    texts = [hit['_source']['对应字段信息'] for hit in hits if '对应字段信息' in hit['_source']]
    metadatas = [{'_id': hit['_id']} for hit in hits if '对应字段信息' in hit['_source']]
    vectordb.add_texts(texts = texts, metadatas = metadatas)
    res = es.scroll(scroll_id = scroll_id, scroll = "1m")
    scroll_id = res['_scroll_id']
    hits = res['hits']['hits']
    count += len(hits)
    if FLAGS.total is not None and count >= FLAGS.total: break

  es.clear_scroll(scroll_id = scroll_id)

if __name__ == "__main__":
  add_options()
  app.run(main)

