import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import pandas as pd
import random

from transformers import AdamW

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_name = option.root_name
root_path = option.root_path

logs_folder = os.path.join(root_path, 'logs', root_name)
save_folder = os.path.join(root_path, 'save', root_name)
sample_folder = os.path.join(root_path, 'sample', root_name)
result_folder = os.path.join(root_path, 'result', root_name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log')
save_path = os.path.join(save_folder, 'best.bin')

logger = get_logger(root_name, logs_path)

from loaders.QALoader import get_loader as get_qa_loader
from modules.QAModule import get_module as get_qa_module

from utils.misc import (
    qa_train, qa_valid, save_checkpoint, load_checkpoint, qa_save_sample
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

train_loader, valid_loader, test_loader = get_qa_loader(option)

logger.info('prepare module')

module = get_qa_module(option).to(device)

logger.info('prepare envs')

parameters = list(module.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
grouped_parameters = [
    {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(grouped_parameters, lr = option.learning_rate)

logger.info('start training')

valid_best_loss = float('inf')
early_stop = 0
for epoch in range(1, option.num_epoch + 1):
    train_info = qa_train(train_loader, module, criterion, optimizer, device)
    valid_info = qa_valid(valid_loader, module, criterion, optimizer, device)
    logger.info(
        'epoch %d: train_loss: %.7f, valid_loss: %.7f' %
        (epoch, train_info['loss'], valid_info['loss'])
    )
    if  valid_best_loss > valid_info['loss']:
        valid_best_loss = valid_info['loss']
        early_stop = 0
        save_checkpoint(save_path, module, optimizer, epoch)
        qa_save_sample (
            sample_folder, 
            valid_info['start_true_fold'],
            valid_info['start_prob_fold'],
            valid_info['start_pred_fold'],
            valid_info['end_true_fold'],
            valid_info['end_prob_fold'],
            valid_info['end_pred_fold']
        )
    else:
        early_stop += 1
    if  early_stop > option.early_stop:
        break

logger.info('start testing')

load_checkpoint(save_path, module, optimizer)

test_info = qa_valid(test_loader, module, criterion, optimizer, device)

logger.info('test_loss: %.7f' % (test_info['loss']))

qa_save_sample(
    result_folder, 
    test_info['start_true_fold'],
    test_info['start_prob_fold'],
    test_info['start_pred_fold'],
    test_info['end_true_fold'],
    test_info['end_prob_fold'],
    test_info['end_pred_fold']
)
