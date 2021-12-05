import os

import torch
import torch.nn as nn

from transformers import BertConfig
from transformers import BertModel, BertForQuestionAnswering

from collections import namedtuple

QuestionAnsweringModelOutput = namedtuple('QuestionAnsweringModelOutput', ['loss', 'start_logits', 'end_logits'])

# Simple implementation of BertForQuestionAnswering. The official implementation is as follows:
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L1788

class MyBertForQuestionAnswering(nn.Module):
    def __init__(self, model_name_or_path, hidden_size, num_classes):
        super(MyBertForQuestionAnswering , self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_name_or_path)
        self.bert_module = BertModel .from_pretrained(model_name_or_path, config = self.bert_config)
        for param in self.bert_module.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.linear  = nn.Linear (hidden_size, num_classes)

    def forward(self, input_ids, input_mask, segment_ids, start_positions = None, end_positions = None):
        output = self.bert_module(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        logits = self.linear(self.dropout(output.last_hidden_state))
        start_logits, end_logits = logits.split(1, dim = -1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if (
            start_positions is not None and
            end_positions is not None
        ):
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss = total_loss,
            start_logits = start_logits,
            end_logits = end_logits
        )

def get_module(option):
    bert_path = option.bert_path
    bert_name = option.bert_name
    model_name_or_path = os.path.join(bert_path, bert_name)
    if option.official:
        config = BertConfig.from_pretrained(model_name_or_path)
        module = BertForQuestionAnswering.from_pretrained(model_name_or_path, config = config)
    else:
        hidden_size = option.hidden_size
        num_classes = option.num_classes
        module = MyBertForQuestionAnswering(model_name_or_path, hidden_size, num_classes)
    return module

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    seq_len = option.max_qsn_len + option.max_psg_len + 3
    batch_size = option.batch_size

    input_ids = torch.zeros((batch_size, seq_len), dtype = torch.long)
    input_mask = torch.zeros((batch_size, seq_len), dtype = torch.long)
    segment_ids = torch.zeros((batch_size, seq_len), dtype = torch.long)
    start_positions = torch.tensor(range(batch_size), dtype = torch.long)
    end_positions = torch.tensor(range(batch_size), dtype = torch.long)

    output = module(input_ids, input_mask, segment_ids)
    print(output.loss) # None
    print(output.start_logits.shape) # (batch_size, seq_len)
    print(output.end_logits.shape)   # (batch_size, seq_len)

    output = module(input_ids, input_mask, segment_ids, start_positions = start_positions, end_positions = end_positions)
    print(output.loss) # Tensor with grad
    print(output.start_logits.shape)  # (batch_size, seq_len)
    print(output.end_logits.shape)    # (batch_size, seq_len)
