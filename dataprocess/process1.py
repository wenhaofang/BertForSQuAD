import os
import subprocess

import json
import pandas as pd

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

sources_path = option.sources_path
targets_path = option.targets_path
dataset_name = option.dataset_name

train_file_path = option.train_file_path
valid_file_path = option.valid_file_path
test_file_path  = option.test_file_path 

# download

subprocess.run('mkdir -p %s' % sources_path, shell = True)

squad1_1 = 'https://data.deepai.org/squad1.1.zip'
squad2_0 = 'https://data.deepai.org/squad2.0.zip'

onweb_path = \
    squad1_1 if dataset_name == 'squad1.1' else \
    squad2_0 if dataset_name == 'squad2.0' else ''

ziped_path = os.path.join(sources_path, dataset_name + '.zip')
unzip_path = os.path.join(sources_path, dataset_name)

if not os.path.exists(ziped_path):
    os.system('wget  %s -O %s' % (onweb_path, ziped_path))

if not os.path.exists(unzip_path):
    os.system('unzip %s -d %s' % (ziped_path, unzip_path))

# process

def read_squad11(file_path):
    result = []
    with open(file_path, 'r', encoding = 'utf-8') as json_file:
        json_data = json.load(json_file)
        for data in json_data['data']:
            title = data['title']
            paras = data['paragraphs']
            for para in paras:
                context = para['context']
                qas = para['qas']
                for qa in qas:
                    qa_id = qa['id']
                    question = qa['question']
                    answer = qa['answers'][0]
                    result.append({
                        'id'          : qa_id,
                        'title'       : title,
                        'text'        : context,
                        'question'    : question,
                        'answer_start': answer['answer_start'],
                        'answer'      : answer['text']
                    })
    return result

def read_squad20(file_path):
    result = []
    with open(file_path, 'r', encoding = 'utf-8') as json_file:
        json_data = json.load(json_file)
        for data in json_data['data']:
            title = data['title']
            paras = data['paragraphs']
            for para in paras:
                context = para['context']
                qas = para['qas']
                for qa in qas:
                    qa_id = qa['id']
                    question = qa['question']
                    answer = qa['answers'][0] if not qa['is_impossible'] else None
                    result.append({
                        'id'          : qa_id,
                        'title'       : title,
                        'text'        : context,
                        'question'    : question,
                        'answer_start': answer['answer_start'] if answer else -1,
                        'answer'      : answer['text'] if answer else ''
                    })
    return result

target_folder = os.path.join(targets_path, dataset_name)

subprocess.run('mkdir -p %s' % (target_folder), shell = True)

train_path = os.path.join(target_folder, train_file_path)
valid_path = os.path.join(target_folder, valid_file_path)
test_path  = os.path.join(target_folder, test_file_path )

if dataset_name == 'squad1.1':
    file_a = os.path.join(unzip_path, 'train-v1.1.json')
    file_b = os.path.join(unzip_path, 'dev-v1.1.json')
    data_a = read_squad11(file_a)
    data_b = read_squad11(file_b)
    radios = int(len(data_a) * 0.9)
    train_data = data_a[:radios]
    valid_data = data_a[radios:]
    test_data  = data_b
    pd.DataFrame(train_data).to_csv(train_path, index = None)
    pd.DataFrame(valid_data).to_csv(valid_path, index = None)
    pd.DataFrame(test_data ).to_csv(test_path , index = None)

if dataset_name == 'squad2.0':
    file_a = os.path.join(unzip_path, 'train-v2.0.json')
    file_b = os.path.join(unzip_path, 'dev-v2.0.json')
    data_a = read_squad20(file_a)
    data_b = read_squad20(file_b)
    radios = int(len(data_a) * 0.9)
    train_data = data_a[:radios]
    valid_data = data_a[radios:]
    test_data  = data_b
    pd.DataFrame(train_data).to_csv(train_path, index = None)
    pd.DataFrame(valid_data).to_csv(valid_path, index = None)
    pd.DataFrame(test_data ).to_csv(test_path , index = None)
