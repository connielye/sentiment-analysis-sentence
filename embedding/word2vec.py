import collections
import tensorflow as tf
import math
import numpy as np


"""
Dataset class buids datasets into training data, validation data, a list of word and its counts in the data, dictionary and reversed dictionary for index look up
"""


class Dataset:

    def __init__(self, input_file, vocabulary_size):
        self.input_file = input_file
        self.vocabulary_size = vocabulary_size
        self.input_data = self.load_data()

    def load_data(self):
        input_data = []
        with open(self.input_file, "r", encoding='utf-8') as file:
            line = file.readline()
            tokens = line.lower().strip().split(' ')
            input_data.append(tokens)
        return input_data

    def build_dictionary(self):
        tokens = []
        tokens.extend(self.input_data)
        dictionary = dict()
        dictionary['UNK'] = 0
        dictionary['START'] = 1
        dictionary['END'] = 2
        word_count = collections.Counter(tokens).most_common(self.vocabulary_size-1)
        for word, counts in word_count:
            dictionary[word] = len(dictionary)
        return word_count, dictionary

    def build_dataset(self):
        word_count, dictionary = self.build_dictionary()
        data = []
        unk_count = 0
        for token_list in self.input_data:
            for word in token_list:
                index = dictionary.get(word, 0)
                data.append(index)
                if index == 0:
                    unk_count += 1
            word_count.append(('UNK', unk_count))
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, word_count, dictionary, reversed_dictionary


"""
skip gram model
"""


class SkipGram:

    def __init__(self, train_data, val_data, reversed_dictionary, vocabulary_size, embedding_size, window_size,
                 batch_size, learning_rate, num_epochs, num_sampled):
        self.train_data = train_data
        self.val_data = val_data
        self.vocabulary_size = vocabulary_size
        self.reversed_dictionary = reversed_dictionary
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_sampled = num_sampled
        self.saver = tf.train.Saver()

    def generate_input_output_pair(self):
        pairs = []
        for token_list in self.train_data:
            for index in range(len(token_list)):
                target_word = self.train_data[index]
                context_words = []
                for num in range(1, self.window_size+1):
                    if index - num < 0:
                        context_words.append(1)
                    elif index + num > 0:
                        context_words.append(2)
                    else:
                        context_words.append(token_list[index - num])
                        context_words.append(token_list[index + num])
                for context_word in context_words:
                    pair = (target_word, context_word)
                    pairs.append(pair)
                    pairs = set(pairs)
        return pairs

    def generate_batch(self):
        pairs = self.generate_input_output_pair()
        batches = []
        batch_number = len(self.train_data)//self.batch_size
        for number in range(batch_number):
            batch = pairs[number*self.batch_size:number*self.batch_size+self.batch_size]
            inputs = []
            outputs = []
            for (target, context) in batch:
                inputs.append(np.eye(target))
                outputs.append(np.eye(context))
            batch_pair = (inputs, outputs)
            batches.append(batch_pair)
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            batch_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, ]) #[batch_size, ]
            batch_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1]) #[batch_size, 1]
            val_dataset = tf.constant(self.val_data, dtype=tf.int32)

        with tf.variable_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs) #[batch_size, embedding_size]
            val_embeddings = tf.nn.embedding_lookup(embeddings, val_dataset) #[batch_size, embedding_size]

        with tf.variable_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                      stddev=1.0/math.sqrt(self.embedding_size))) #[vocab_size, embedding_size]

        with tf.variable_scope('biases'):
            biases = tf.Variable(tf.zeros([self.vocabulary_size])) #[batch_size, 1]

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights, biases=biases, labels=batch_labels,
                                                 inputs=batch_embeddings, num_sampled=self.num_sampled,
                                                 num_classes=self.vocabulary_size))

        with tf.name_scope('norm'):
            norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings/norm

        with tf.name_scope('similarity'):
            similarity = tf.matmul(val_embeddings, normalized_embeddings, transpose_b=True)

        return batch_inputs, batch_labels, normalized_embeddings, loss, similarity

    def train(self):
        batches = self.generate_batch()
        batch_inputs, batch_labels, normalized_embeddings, loss, similarity = self.model()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for epoch in range(self.num_epochs):
                print("Number of epochs: ", epoch+1)
                average_loss = 0.0
                for iteration, batch_data in enumerate(batches):
                    (inputs, labels) = batch_data
                    feed_dict = {batch_inputs: inputs, batch_labels: labels}

                    _, loss_val = session.run([optimizer, loss], feed_dict)
                    average_loss += loss_val

                    if iteration % 1000 == 0:
                        if iteration > 0:
                            average_loss /= 1000
                        print('Loss at iteration ', iteration, ": ", average_loss)
                        average_loss = 0

                    if iteration % 5000 == 0:
                        sim = similarity.eval()
                        for i in range(len(self.val_data)):
                            top_10 = 10
                            nearest = (-sim[i, :]).argsort()[0:top_10+1]
                            for nw in nearest:
                                print(self.reversed_dictionary[nw], end=' ', flush=True)


"""
continuous bag-of-word 
"""


class CBoW:

    def __init__(self, train_data, val_data, reversed_dictionary, vocabulary_size, embedding_size, window_size,
                 batch_size, learning_rate, num_epochs, num_sampled):
        self.train_data = train_data
        self.val_data = val_data
        self.vocabulary_size = vocabulary_size
        self.reversed_dictionary = reversed_dictionary
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_sampled = num_sampled
        self.saver = tf.train.Saver()

    def generate_input_output_pair(self):
        pairs = []
        for token_list in self.train_data:
            for index in range(len(token_list)):
                target_word = self.train_data[index]
                context_words = []
                for num in range(1, self.window_size+1):
                    if index - num < 0:
                        context_words.append(1)
                    elif index + num > 0:
                        context_words.append(2)
                    else:
                        context_words.append(self.train_data[index - num])
                        context_words.append(self.train_data[index + num])
                pair = (context_words, target_word)
                pairs.append(pair)
        return pairs

    def generate_batch(self):
        pairs = self.generate_input_output_pair()
        batches = []
        batch_number = len(self.train_data)//self.batch_size
        for number in range(batch_number):
            batch = pairs[number*self.batch_size:number*self.batch_size+self.batch_size]
            inputs = []
            outputs = []
            for (context_words, target_word) in batch:
                inputs.append(context_words)
                outputs.append(target_word)
            batch_pair = (inputs, outputs)
            batches.append(batch_pair)
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            batch_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, ])
            batch_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            val_dataset = tf.constant(self.val_data, dtype=tf.int32)

        with tf.variable_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)
            val_embeddings = tf.nn.embedding_lookup(embeddings, val_dataset)

        with tf.variable_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                      stddev=1.0/math.sqrt(self.embedding_size)))

        with tf.variable_scope('biases'):
            biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights, biases=biases, labels=batch_labels,
                                                 inputs=batch_embeddings, num_sampled=self.num_sampled,
                                                 num_classes=self.vocabulary_size))

        with tf.name_scope('norm'):
            norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings/norm

        with tf.name_scope('similarity'):
            similarity = tf.matmul(val_embeddings, normalized_embeddings, transpose_b=True)

        return batch_inputs, batch_labels, normalized_embeddings, loss, similarity

    def train(self):
        batches = self.generate_batch()
        batch_inputs, batch_labels, normalized_embeddings, loss, similarity = self.model()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for epoch in range(self.num_epochs):
                print("Number of epochs: ", epoch+1)
                average_loss = 0.0
                for iteration, batch_data in enumerate(batches):
                    inputs, labels = batch_data
                    feed_dict = {batch_inputs: inputs, batch_labels: labels}

                    _, loss_val = session.run([optimizer, loss], feed_dict)
                    average_loss += loss_val

                    if iteration % 1000 == 0:
                        if iteration > 0:
                            average_loss /= 1000
                        print('Loss at iteration ', iteration, ": ", average_loss)
                        average_loss = 0

                    if iteration % 5000 == 0:
                        sim = similarity.eval()
                        for i in range(len(self.val_data)):
                            top_10 = 10
                            nearest = (-sim[i, :]).argsort()[0:top_10+1]
                            for nw in nearest:
                                print(self.reversed_dictionary[nw], end=' ', flush=True)
