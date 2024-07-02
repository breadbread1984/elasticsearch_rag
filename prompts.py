#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def elasticsearch_template(tokenizer):
  PROMPT_SUFFIX = """Only use the following Elasticsearch indices:
{indices_info}

Question: {input}
ESQuery:"""

  DEFAULT_DSL_TEMPLATE = """Given an input question, create a syntactically correct Elasticsearch query to run. Always limit your query to at most {top_k} results, unless the user specifies in their question a specific number of examples they wish to obtain, or unless its implied that they want to see all. You can order the results by a relevant column to return the most interesting examples in the database.

Unless told to do not query for all the columns from a specific index, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index. Return the query as valid json.

Use the following format:

Question: Question here
ESQuery: Elasticsearch Query formatted as json
"""

  messages = [
    {'role': 'system', 'content': DEFAULT_DSL_TEMPLATE},
    {'role': 'user', 'content': PROMPT_SUFFIX}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template =  PromptTemplate.from_template(template = prompt, input_variables = ['top_k', 'indices_info', 'input'])
  return template
