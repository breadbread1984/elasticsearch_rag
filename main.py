#!/usr/bin/python3

from absl import app, flags
from prompts import query_template, answer_template
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from models import Llama3, CodeLlama, Qwen2, Qwen2_TP, CodeQwen1_5, Qwen1_5

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama3', 'codellama', 'qwen2', 'qwen2_tp', 'codeqwen', 'qwen1.5'}, help = 'available models')
  flags.DEFINE_string('host', default = None, help = 'elastic search host')
  flags.DEFINE_string('index', default = 'qd_asset', help = 'index')
  flags.DEFINE_string('username', default = 'elastic', help = 'username')
  flags.DEFINE_string('password', default = None, help = 'password')
  flags.DEFINE_integer('top_k', default = 3, help = 'topk')
  flags.DEFINE_boolean('locally', default = False, help = 'whether to run locally')

def main(unused_argv):
  tokenizer, llm = {
    'llama3': Llama3,
    'codellama': CodeLlama,
    'qwen2': Qwen2,
    'qwen2_tp': Qwen2_TP,
    'codeqwen': CodeQwen1_5,
    'qwen1.5': Qwen1_5}[FLAGS.model](FLAGS.locally)
  host_with_authentication = FLAGS.host[:FLAGS.host.find('://') + 3] + FLAGS.username + ":" + FLAGS.password + "@" + FLAGS.host[FLAGS.host.find('://') + 3:]
  es = Elasticsearch(host_with_authentication)
  embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Chroma(embedding_function = embeddings, persist_directory = 'db')
  retriever = vectordb.as_retriever()
  while True:
    query = input('要问什么问题呢？>')
    docs = retriever.get_relevant_documents(query)
    ids = {doc.metadata['_id'] for doc in docs}
    res = es.search(index = FLAGS.index, scroll = '1m', size = 10, body = {"query": "terms": {"_id": list(ids)}})
    print(res)

if __name__ == "__main__":
  add_options()
  app.run(main)

