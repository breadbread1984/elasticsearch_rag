#!/usr/bin/python3

from absl import app, flags
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from chains import rephrase_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = None, help = 'elastic search host')
  flags.DEFINE_string('index', default = 'qd_asset', help = 'index')
  flags.DEFINE_string('username', default = 'elastic', help = 'username')
  flags.DEFINE_string('password', default = None, help = 'password')

def main(unused_argv):
  host_with_authentication = FLAGS.host[:FLAGS.host.find('://') + 3] + FLAGS.username + ":" + FLAGS.password + "@" + FLAGS.host[FLAGS.host.find('://') + 3:]
  es = Elasticsearch(host_with_authentication)
  embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Chroma(embedding_function = embeddings, persist_directory = 'db')
  retriever = vectordb.as_retriever()
  rephrase_chain_ = rephrase_chain()
  while True:
    query = input('要问什么问题呢？>')
    docs = retriever.get_relevant_documents(query)
    ids = {doc.metadata['_id'] for doc in docs}
    canonical = rephrase_chain_.invoke({'context': query})
    if canonical.startswith('Assistant: '): canonical = canonical.replace('Assistant: ','')
    docs2 = retriever.get_relevant_documents(canonical)
    ids2 = {doc.metadata['_id'] for doc in docs2}
    ids = ids.union(ids2)
    res = es.search(index = FLAGS.index, scroll = '1m', body = {"query": {"terms": {"_id": list(ids)}}})
    print(res)

if __name__ == "__main__":
  add_options()
  app.run(main)

