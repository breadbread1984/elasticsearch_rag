# Introduction

this repo implement QA system interactive with elasticsearch database

# Usage

## Install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## Run QA system

```shell
python3 main.py --model (llama3|codellama|qwen2|codeqwen|qwen1.5) --host <host> --username <username> --password <password> [--top_k <number of matches>] [--locally]
```

## Example

```shell
CUDA_VISIBLE_DEVICES=0 python3 main.py --host http://es8.shuidata.cn:88 --password ***** --model qwen2 --locally
```
