#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def rephrase_template(tokenizer):
  system_message = """提供一句话，请把这句话转义为常用标准表达"""
  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "{context}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template

