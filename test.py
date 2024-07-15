#!/usr/bin/python3

from models import Llama3
from prompts import rephrase_template

tokenizer, llm = Llama3(False)
template = rephrase_template(tokenizer)
chain = template | llm

print(chain.invoke({'context': "抄表数据"}))
print(chain.invoke({'context': '水表读数'}))
print(chain.invoke({'context': '抄水量'}))
