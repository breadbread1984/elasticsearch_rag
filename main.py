#!/usr/bin/python3

from absl import app, flags
from prompts import elasticsearch_template
from models import Llama3, CodeLlama, Qwen2, CodeQwen1_5, Qwen1_5

def add_options():
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama3', 'codellama', 'qwen2', 'codeqwen', 'qwen1.5'}, help = 'available models')
  flags.DEFINE_string('host', default = None, help = 'elastic search host')
  flags.DEFINE_string('index', default = 'qd_asset', help = 'index')
  flags.DEFINE_string('username', default = 'elastic', help = 'username')
  flags.DEFINE_string('password', default = None, help = 'password')
  flags.DEFINE_integer('top_k', default = 3, help = 'topk')
  flags.DEFINE_boolean('locally', default = False, help = 'whether to run locally')

def main():
  tokenizer, llm = {
    'llama3': Llama3,
    'codellama': CodeLlama,
    'qwen2': Qwen2,
    'codeqwen': CodeQwen1_5,
    'qwen1.5': Qwen1_5}[FLAGS.model](FLAGS.locally)
  chain = elasticsearch_template(tokenizer) | llm
  while True:
    query = input('要问什么问题呢？>')
    response = chain.invoke({'top_k': FLAGS.top_k, 'indices_info': FLAGS.index, 'input': query})
    print(response)

if __name__ == "__main__":
  add_options()
  app.run(main)
