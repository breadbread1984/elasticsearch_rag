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

