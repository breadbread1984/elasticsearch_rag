#!/usr/bin/python3

from models import Qwen2
from prompts import rephrase_template

def rephrase_chain():
  tokenizer, llm = Qwen2(True)
  template = rephrase_template(tokenizer)
  chain = template | llm
  return chain

