### Introduction

This is a repository for Extractive Question Answering task, using BERT model and SQuAD dataset.

### Environment

```shell
# create environment
conda create --name NAME python=3.8
# activate environment
conda activate NAME
# install packages
pip install -r requirements.txt
```

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

### Unit Test

* for loader

```shell
# QALoader for SQuAD1.1
PYTHONPATH=./ python loaders/QALoader.py --dataset_name squad1.1
# QALoader for SQuAD2.0
PYTHONPATH=./ python loaders/QALoader.py --dataset_name squad2.0
```

* for module

```shell
# QAModule
# You can add --official to use official implementation of BertForQuestionAnswering
PYTHONPATH=./ python modules/QAModule.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`
