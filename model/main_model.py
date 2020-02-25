import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.python.lib.io import file_io
import argparse

from cnn import CNN
from bi_LSTM import BiLSTM



def train_model_cnn(train_file, saved_file):

    with file_io.FileIO(train_file, 'r') as input_file:
        input_data = pd.read_csv(input_file, sep=';', engine='python', header=None, names=['Index', 'Text', 'Label'])

    df = pd.DataFrame(input_data)

    text = pd.Series(df['Text'])
    label = pd.Series(df['Label'])

    token_list, tokens = [], []
    for row in text:
        row_string = str(row)
        token = re.split("\W+", row_string)
        token_list.append(token)
        tokens.extend(token)

    dictionary = {word: i for i, word in enumerate(set(tokens))}
    dictionary["PAD"] = len(dictionary)
    reversed_dictionary = {i: word for i, word in enumerate(set(tokens))}
    reversed_dictionary[len(dictionary)] = "PAD"

    vocab_size = len(dictionary)

    inputs = []
    for list in token_list:
        if len(list) < 128:
            for i in range(len(list), 128):
                list.append("PAD")
        inputs.append(np.asarray([dictionary[word] for word in list]))

    targets = []
    val_targets = []
    for row in label:
        target = int(row)
        targets.append(np.eye(3)[target])
        val_targets.append(target)

    sep_index = int(len(inputs)*0.8)
    train_set = []
    for i in range(sep_index):
        input = inputs[i]
        target = targets[i]
        train_set.append((input, target))

    val_set = []
    for i in range(sep_index, len(inputs)):
        input = inputs[i]
        target = val_targets[i]
        val_set.append((input, target))

    model = CNN(train_set, val_set, 0.01, 1000, 3, 1, 300, 128, [1,2,3], 3, vocab_size, saved_file)
    model.train()


def train_model_lstm(train_file):
    input_data = pd.read_csv(train_file, sep=';', engine='python', header=None, names=['Index', 'Text', 'Label'])
    df = pd.DataFrame(input_data)

    text = pd.Series(df['Text'])
    label = pd.Series(df['Label'])

    token_list, tokens = [], []
    for row in text:
        row_string = str(row)
        token = re.split("\W+", row_string)
        token_list.append(token)
        tokens.extend(token)

    dictionary = {word: i for i, word in enumerate(set(tokens))}
    dictionary["PAD"] = len(dictionary)
    dictionary["UNK"] = len(dictionary)
    reversed_dictionary = {i: word for i, word in enumerate(set(tokens))}
    reversed_dictionary[len(dictionary)] = "PAD"
    reversed_dictionary[len(dictionary)] = "UNK"

    vocab_size = len(dictionary)

    inputs = []
    for list in token_list:
        if len(list) < 128:
            for i in range(len(list), 128):
                list.append("PAD")
        inputs.append(np.asarray([dictionary[word] for word in list]))

    targets = []
    val_targets = []
    for row in label:
        target = int(row)
        targets.append(np.eye(3)[target])
        val_targets.append(target)

    sep_index = int(len(inputs)*0.8)
    train_set = []
    for i in range(sep_index):
        input = inputs[i]
        target = targets[i]
        train_set.append((input, target))

    val_set = []
    for i in range(sep_index, len(inputs)):
        input = inputs[i]
        target = val_targets[i]
        val_set.append((input, target))

    model = BiLSTM(train_set, val_set, 0.01, 1000, 128, 3, 1, 50, 128, vocab_size)
    model.train()

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', help='GCS or local paths to training data', required=True)
    parser.add_argument('--job-dir', help='GCS location to save model', required=True)

    args = parser.parse_args()
    arguments = args.__dict__

    train_model_cnn(**arguments)


train_model_cnn('/home/bingxin/Documents/trainingdata/test.csv','/home/bingxin/Documents/tmp/saved_model')
"""