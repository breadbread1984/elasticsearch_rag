#!/usr/bin/python3

from os import environ
from huggingface_hub import login
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import VLLM
import config

def Llama3(locally = False):
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
      task = "text-generation",
      device = 0,
      pipeline_kwargs = {
        "max_length": 8192,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = config.huggingface_token
    llm = HuggingFaceEndpoint(
      endpoint_url = "meta-llama/Meta-Llama-3-8B-Instruct",
      task = "text-generation",
      do_sample = True,
      temperature = 0.6,
      top_p = 0.9,
    )
  return tokenizer, llm

def CodeLlama(locally = False):
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 16384,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = config.huggingface_token
    llm = HuggingFaceEndpoint(
      endpoint_url = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      do_sample = True,
      temperature = 0.8,
      top_p = 0.8,
      use_cache = True
    )
  return tokenizer, llm

def Qwen2(locally = False):
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'Qwen/Qwen2-7B-Instruct',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.41.2",
        "use_cache": True,
        "max_length": 131072,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = config.huggingface_token
    llm = HuggingFaceEndpoint(
      endpoint_url = 'Qwen/Qwen2-7B-Instruct',
      task = 'text-generation',
      do_sample = True,
      temperature = 0.8,
      top_p = 0.8,
    )
  return tokenizer, llm

def Qwen2_TP(locally = False, tp_num = 2):
  assert locally == True, "vllm can only run locally!"
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
  llm = VLLM(model = "Qwen/Qwen2-7B-Instruct", tensor_parallel_size = tp_num, trust_remote_code = True)
  return tokenizer, llm

def CodeQwen1_5(locally = False):
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/CodeQwen1.5-7B-Chat')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'Qwen/CodeQwen1.5-7B-Chat',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 65536,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = config.huggingface_token
    llm = HuggingFaceEndpoint(
      endpoint_url = 'Qwen/CodeQwen1.5-7B-Chat',
      task = 'text-generation',
      do_sample = True,
      temperature = 0.8,
      top_p = 0.8,
    )
  return tokenizer, llm

def Qwen1_5(locally = False):
  login(token = config.huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-14B-Chat')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'Qwen/Qwen1.5-14B-Chat',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 32768,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False,
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = config.huggingface_token
    llm = HuggingFaceEndpoint(
      endpoint_url = 'Qwen/Qwen1.5-14B-Chat',
      task = 'text-generation',
      do_sample = True,
      temperature = 0.8,
      top_p = 0.8,
    )
  return tokenizer, llm
