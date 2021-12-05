import os
import torch
import pandas as pd

from transformers import BertTokenizer

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from dataclasses import dataclass

@dataclass
class InputExample:
    guid: str
    tokens: list
    question: str
    answer: str
    answer_start_pos: int
    answer_end_pos: int

@dataclass
class InputFeature:
    guid: int
    input_ids: list
    input_mask: list
    segment_ids: list
    start_pos: int
    end_pos: int

def read_examples(file_path):
    data_frame = pd.read_csv(file_path)
    examples = []
    for (
        ix, text, answer, question, answer_start
    ) in zip (
        data_frame['id'].tolist(),
        data_frame['text'].tolist(),
        data_frame['answer'].tolist(),
        data_frame['question'].tolist(),
        data_frame['answer_start'].tolist()
    ):
        cur_token = ''
        all_token = []
        char_to_token = []
        for char in text:
            char_to_token.append(len(all_token))
            if (
                char == ' '  or
                char == '\r' or
                char == '\t' or
                char == '\n' or
                ord(char) == 0x202F
            ):
                all_token.append(cur_token)
                cur_token = ''
            else:
                cur_token += char
        if cur_token != '':
            all_token.append(cur_token)

        if (
            answer_start == -1 or
            pd.isna(answer)
        ):
            answer_start_pos = -1
            answer_end_pos = -1
        else:
            answer_start_pos = char_to_token[answer_start]
            answer_end_pos = char_to_token[answer_start + len(answer) - 1]

        examples.append(InputExample(
            ix, all_token, question, answer, answer_start_pos, answer_end_pos
        ))

    return examples

def convert_examples_to_features(examples, tokenizer, max_qsn_len, max_psg_len):
    CLS_TOKEN = tokenizer.cls_token
    SEP_TOKEN = tokenizer.sep_token
    PAD_TOKEN = tokenizer.pad_token
    UNK_TOKEN = tokenizer.unk_token

    features = []
    for example_idx, example in enumerate(examples):

        question_tokens = tokenizer.tokenize(example.question)

        paragraph_tokens = []
        tok_to_ori_index = []
        ori_to_tok_index = []
        for idx, token in enumerate(example.tokens):
            ori_to_tok_index.append(len(paragraph_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_ori_index.append(idx)
                paragraph_tokens.append(sub_token)

        if (
            example.answer_start_pos == -1 or
            example.answer_end_pos == -1
        ):
            token_start_pos = -1
            token_end_pos = -1
        else:
            token_start_pos = ori_to_tok_index[example.answer_start_pos]
            if (
                example.answer_end_pos < len(example.tokens) - 1
            ):
                token_end_pos = ori_to_tok_index[example.answer_end_pos + 1]
            else:
                token_end_pos = len(paragraph_tokens) - 1

            should_break = False
            answer = ' '.join(tokenizer.tokenize(example.answer))
            for new_start in range(token_start_pos, token_end_pos + 1):
                if should_break:
                    break
                for new_end in range(token_end_pos, new_start - 1, -1):
                    if should_break:
                        break
                    text_span = ' '.join(paragraph_tokens[new_start:(new_end + 1)])
                    if text_span == answer:
                        token_start_pos = new_start
                        token_end_pos = new_end
                        should_break = True

        tokens = []
        segment_ids = []

        tokens.append(CLS_TOKEN)
        segment_ids.append(0)

        qsn_len = 0
        for token in question_tokens:
            if len(tokens) - 1 < max_qsn_len:
                qsn_len += 1
                tokens.append(token)
                segment_ids.append(0)

        tokens.append(SEP_TOKEN)
        segment_ids.append(0)

        psg_len = 0
        for token in paragraph_tokens:
            if len(tokens) - 2 < max_qsn_len + max_psg_len:
                psg_len += 1
                tokens.append(token)
                segment_ids.append(1)

        tokens.append(SEP_TOKEN)
        segment_ids.append(1)

        input_mask = [1] * len(tokens)

        while len(tokens) < (max_qsn_len + max_psg_len + 3):
            tokens.append(PAD_TOKEN)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        while len(input_mask) < (max_qsn_len + max_psg_len + 3):
            input_mask.append(0)
            segment_ids.append(0)

        doc_start = 0
        doc_end = doc_start + psg_len - 1
        if (
            token_start_pos < doc_start or
            token_end_pos > doc_end
        ):
            start_pos = 0
            end_pos = 0
        else:
            doc_offset = qsn_len + 2
            start_pos = token_start_pos + doc_offset
            end_pos = token_end_pos + doc_offset

        features.append(InputFeature(
            example_idx, input_ids, input_mask, segment_ids, start_pos, end_pos
        ))

    return features

def get_loader(option):
    module_name_or_path = os.path.join(option.bert_path, option.bert_name)

    tokenizer = BertTokenizer.from_pretrained(module_name_or_path, do_lower_case = True)

    train_examples = read_examples(os.path.join(option.targets_path, option.dataset_name, option.train_file_path))
    valid_examples = read_examples(os.path.join(option.targets_path, option.dataset_name, option.valid_file_path))
    test_examples  = read_examples(os.path.join(option.targets_path, option.dataset_name, option.test_file_path ))

    train_features = convert_examples_to_features(train_examples, tokenizer, option.max_qsn_len, option.max_psg_len)
    valid_features = convert_examples_to_features(valid_examples, tokenizer, option.max_qsn_len, option.max_psg_len)
    test_features  = convert_examples_to_features(test_examples , tokenizer, option.max_qsn_len, option.max_psg_len)

    train_input_ids   = torch.tensor([feature.input_ids   for feature in train_features], dtype = torch.long)
    train_input_mask  = torch.tensor([feature.input_mask  for feature in train_features], dtype = torch.long)
    train_segment_ids = torch.tensor([feature.segment_ids for feature in train_features], dtype = torch.long)
    train_start_pos   = torch.tensor([feature.start_pos   for feature in train_features], dtype = torch.long)
    train_end_pos     = torch.tensor([feature.end_pos     for feature in train_features], dtype = torch.long)
    train_guid        = torch.tensor([feature.guid        for feature in train_features], dtype = torch.long)

    valid_input_ids   = torch.tensor([feature.input_ids   for feature in valid_features], dtype = torch.long)
    valid_input_mask  = torch.tensor([feature.input_mask  for feature in valid_features], dtype = torch.long)
    valid_segment_ids = torch.tensor([feature.segment_ids for feature in valid_features], dtype = torch.long)
    valid_start_pos   = torch.tensor([feature.start_pos   for feature in valid_features], dtype = torch.long)
    valid_end_pos     = torch.tensor([feature.end_pos     for feature in valid_features], dtype = torch.long)
    valid_guid        = torch.tensor([feature.guid        for feature in valid_features], dtype = torch.long)

    test_input_ids    = torch.tensor([feature.input_ids   for feature in test_features ], dtype = torch.long)
    test_input_mask   = torch.tensor([feature.input_mask  for feature in test_features ], dtype = torch.long)
    test_segment_ids  = torch.tensor([feature.segment_ids for feature in test_features ], dtype = torch.long)
    test_start_pos    = torch.tensor([feature.start_pos   for feature in test_features ], dtype = torch.long)
    test_end_pos      = torch.tensor([feature.end_pos     for feature in test_features ], dtype = torch.long)
    test_guid         = torch.tensor([feature.guid        for feature in test_features ], dtype = torch.long)

    train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, train_guid)
    valid_dataset = TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_start_pos, valid_end_pos, valid_guid)
    test_dataset  = TensorDataset(test_input_ids , test_input_mask , test_segment_ids , test_start_pos , test_end_pos , test_guid )

    train_dataloader = DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True )
    valid_dataloader = DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = False)
    test_dataloader  = DataLoader(test_dataset , batch_size = option.batch_size, shuffle = False)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    # timing: 1450s for SQuAD1.1, 2100s for SQuaD2.0
    train_loader, valid_loader, test_loader = get_loader(option)

    for mini_batch in train_loader:
        input_ids, input_mask, segment_ids, start_pos, end_pos, guid = mini_batch
        print(input_ids.shape)   # (batch_size, max_seq_len)
        print(input_mask.shape)  # (batch_size, max_seq_len)
        print(segment_ids.shape) # (batch_size, max_seq_len)
        print(start_pos.shape)   # (batch_size)
        print(end_pos.shape)     # (batch_size)
        print(guid.shape)        # (batch_size)
        break
