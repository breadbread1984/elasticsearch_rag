#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def rephrase_template(tokenizer):
  system_message = """Given a sentence, please rephrase it to a canonical expression form in the same language"""
  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "{context}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template

