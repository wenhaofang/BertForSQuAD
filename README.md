### Introduction

This is a repository for Extractive Question Answering task, using BERT model and SQuAD dataset.

### Data Process

* process0.py

```shell
# download BERT: bert-base-cased
PYTHONPATH=./ python dataprocess/process0.py --bert_name bert-base-cased
# download BERT: bert-base-uncased
PYTHONPATH=./ python dataprocess/process0.py --bert_name bert-base-uncased
```

* process1.py

```shell
# download and process SQuAD1.1
PYTHONPATH=./ python dataprocess/process1.py --dataset_name squad1.1
# download and process SQuAD2.0
PYTHONPATH=./ python dataprocess/process1.py --dataset_name squad2.0
```
