import os
import subprocess

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

bert_path = option.bert_path
bert_name = option.bert_name

target_folder = os.path.join(bert_path, bert_name)

subprocess.run('mkdir -p %s' % (target_folder), shell = True)

file_list = ['config.json', 'pytorch_model.bin', 'vocab.txt']

for file_name in file_list:

    base_url = f'https://huggingface.co/{bert_name}/resolve/main/{file_name}'

    os.system('wget -P %s %s' % (target_folder, base_url))
