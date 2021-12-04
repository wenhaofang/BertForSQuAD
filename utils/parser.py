import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # basic

    parser.add_argument('--root_name', default = 'main', help = '')
    parser.add_argument('--root_path', default = 'result', help = '')

    # loader

    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')
    parser.add_argument('--dataset_name', default = 'squad1.1', choices = ['squad1.1', 'squad2.0'], help = '')

    parser.add_argument('--train_file_path', default = 'train.csv', help = '')
    parser.add_argument('--valid_file_path', default = 'valid.csv', help = '')
    parser.add_argument('--test_file_path' , default = 'test.csv' , help = '')

    # module

    parser.add_argument('--bert_path', default = 'bert', help = '')
    parser.add_argument('--bert_name', default = 'bert-base-cased', choices = ['bert-base-cased', 'bert-base-uncased'], help = '')

    parser.add_argument('--official', action = 'store_true', help = '')

    parser.add_argument('--hidden_size', type = int, default = 768, help = '')
    parser.add_argument('--num_classes', type = int, default = 2, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
