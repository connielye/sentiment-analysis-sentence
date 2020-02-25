
import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
import re

"""
WordSentimentPrior class represent sentiment lexicon with their ratio positive: 1/alpha, neutral: 1, negative alpha. 
Alpha is the contrast ratio from 0 to 1. If the training words are not in the sentiment lexicon will be treated as 
neutral. 
"""


class WordSentimentPrior:

    def __int__(self, data, input_file, alpha):
        self.data = data
        self.input_file = input_file
        self.alpha = alpha


    def word_senti_ratio(self):
        positive_ratio = 1/self.alpha
        negative_ratio = self.alpha
        df = pd.DataFrame(self.data, ';')
        words = df[:, 0:1]
        scores = df[:, 1:2]
        ratios = []
        sentiment_dictionary = {}
        reversed_sentiment_dictionary = {}
        for i in range(df.size):
            word = words[i]
            sentiment_dictionary[word] = i
            reversed_sentiment_dictionary[i] = word
            score = scores[i]
            ratio = 1
            if score > 0:
                ratio = positive_ratio
            elif score < 0:
                ratio = negative_ratio
            ratios.append(ratio)

        return sentiment_dictionary, reversed_sentiment_dictionary, ratios

    def build_dictionary(self):
        data = self.load_data(self.input_file)
        df = pd.DataFrame(data).copy()
        tokens = df['Tokens']
        sentiment_dictionary, reversed_sentiment_dictionary, ratios = self.word_senti_ratio()
        sentiment_words = sentiment_dictionary.keys()
        dictionary = {}
        words = dictionary.keys()
        word_senti_ratios = []
        for i in range(df.size):
            word_list = tokens[i].strip().split(' ')
            for word in word_list:
                if word not in words:
                    dictionary[word] = len(dictionary)
                if word in sentiment_words:
                    index = sentiment_dictionary[word]
                    ratio = ratios[index]
                    word_senti_ratios.append(ratio)
                else:
                    word_senti_ratios.append(1)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reversed_dictionary, word_senti_ratios


"""
DocSentimentPrior class represents sentiment distribution in the labeled document. Each word from the training documents 
has a vector to present its sentiment distribution:
[positive counts/total counts, neutral counts/total counts, negative counts/total counts]
and also has the ratio positive/alpha * negative counts. If negative counts == 0 or positive == 0, the ratio is the same
as the sentiment lexicon one. If both are zero, the ratio is 1. 
"""


class DocSentimentPrior:

    def __init__(self, tokens, scores, alpha):
        self.tokens = tokens
        self.scores = scores
        self.alpha = alpha

    def word_senti_info(self):
        word_senti_counts = {}
        word_senti_info = []
        word_senti_ratios = []
        dictionary = {}
        dictionary['START'] = 0
        dictionary['END'] = 1
        word_senti_counts['START'] = [0,0,0]
        word_senti_counts['END'] = [0,0,0]
        for i in range(len(self.scores)):
            words = self.tokens[i]
            score = self.scores[i]
            for word in words:
                keys = list(word_senti_counts.keys())
                if word not in keys:
                    word_senti_counts[word] = [0, 0, 0]
                senti_counts = word_senti_counts[word]
                if score >= 7:
                    senti_counts[0] += 1
                elif score <= 4:
                    senti_counts[2] += 1
                else:
                    senti_counts[1] += 1
                """
                if score > 3:
                    senti_counts[0] += 1
                elif score == 3:
                    senti_counts[1] += 1
                else:
                    senti_counts[2] += 1
                """
        for key, value in enumerate(list(word_senti_counts.keys())):
            dictionary[value] = key
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        for word, senti_counts in word_senti_counts.items():
            if word == 'START' or word == 'END':
                word_senti_info.append([0,0,0])
                word_senti_ratios.append(0)
                continue
            total_count = sum(senti_counts)
            senti_info = []
            for senti_count in senti_counts:
                ratio = senti_count / total_count
                senti_info.append(ratio)
            senti_info_vector = np.array(senti_info)
            word_senti_info.append(senti_info_vector)
            positive_count = senti_counts[0]
            negative_count = senti_counts[2]
            if positive_count == 0 and negative_count != 0:
                ratio = 1 / self.alpha
            elif positive_count != 0 and negative_count == 0:
                ratio = self.alpha
            elif positive_count != 0 and negative_count != 0:
                ratio = positive_count / (self.alpha * negative_count)
            else:
                ratio = 1
            word_senti_ratios.append(ratio)
        return dictionary, reversed_dictionary, word_senti_info, word_senti_ratios


def main(data_file, output_file):

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
    dictionary, reversed_dictionary, word_senti_info, word_senti_ratios = doc_prior.word_senti_info()
    with open(output_file, 'w') as output:
        for index, ratio in enumerate(word_senti_ratios):
            word = reversed_dictionary[index]
            output.write(word)
            output.write('\t')
            output.write(ratio)
            output.write('\n')
        output.close()


if __name__ == '__main__':

    main(data_file='testing_data/test.txt', output_file='/home/bingxin/Documents/test/')