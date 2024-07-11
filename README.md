# Introduction

this repo implement QA system interactive with elasticsearch database

# Usage

## Install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## Create index

```shell
python3 create_vectordb.py --host <host> --username <username> --password <password>
```

## Run QA system

```shell
python3 main.py --host <host> --username <username> --password <password>
```

## Example

```shell
CUDA_VISIBLE_DEVICES=0 python3 main.py --host http://es8.shuidata.cn:88 --password *****
```
