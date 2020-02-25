import tensorflow as tf
from tensorflow.python.lib.io import file_io
import re
import argparse
from prior_sentiment_knowledge import DocSentimentPrior
from gloVe import Coocurrence
from sentiment_word_vector import BuildDataset, WLJT_DLJT2, WLJC_DLJC2


def train_model(job_dir = '', data_file='testing_data/test1.txt'):

    logs_path = job_dir + '/logs/'

    with file_io.FileIO(data_file, 'r') as input_file:
        lines = input_file.readlines()
    token_list, scores = [], []
    for line in lines:
        parts = line.split(";;")
        score = int(parts[1])
        text = parts[0]
        tokens = re.split("\W+", text)
        token_list.append(tokens)
        scores.append(score)

    doc_prior = DocSentimentPrior(token_list, scores, 0.3)
    dictionary, reversed_dictionary, word_senti_info, word_senti_ratio = doc_prior.word_senti_info()
    voc_size = len(dictionary)

    matrix_builder = Coocurrence(data_file, 3, 1)
    _, _, dataset = matrix_builder.build_dataset()
    coocurrence_matrix, max_coocurrence = matrix_builder.build_coocurrence()

    data_builder = BuildDataset(dataset, dictionary, reversed_dictionary, coocurrence_matrix, 3, word_senti_info, word_senti_ratio)
    dl_dataset = data_builder.build_dl_dataset()
    data_size = len(dl_dataset)

    """ 
    target, context, coocurrence, target_senti_info, target_senti_ratio, context_senti_info, context_senti_ratio = dl_dataset[32]
    print(coocurrence, target_senti_info, target_senti_ratio, context_senti_info, context_senti_ratio)
    """


    train_set = dl_dataset[:int(data_size*0.8)]
    #eval_set = dl_dataset[int(data_size*0.8):]

    dl_context_model = WLJC_DLJC2(train_set, 50, voc_size, max_coocurrence-10, 0.75, 100, 0.5, 10, reversed_dictionary)
    dl_context_model.train()


if __name__ == '__main__':
    train_model()

"""
    parser = argparse.ArgumentParser()
     Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local path to training data',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    argments = args.__dict__
"""


