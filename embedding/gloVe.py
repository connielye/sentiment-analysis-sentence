import numpy as np
import re
import tensorflow as tf

"""
calculate co_ocurrence matrix from a large corpus
"""


class Coocurrence:

    def __init__(self, input_file, window_size, min_coocurrence):
        self.input_file = input_file
        self.window_size = window_size
        self.min_coocurrence = min_coocurrence

    def load_data(self):
        input_data = []
        with open(self.input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            text = line.split(";;")[0]
            tokens = re.split("\W+", text)
            input_data.append(tokens)
        return input_data

    def build_dataset(self):
        tokens = []
        dataset = []
        input_data = self.load_data()
        for input in input_data:
            tokens.extend(input)
        dictionary = dict()
        dictionary['START'] = 0
        dictionary['END'] = 1
        for token in tokens:
            words = list(dictionary.keys())
            if token not in words:
                dictionary[token] = len(dictionary)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        for token_list in input_data:
            index_list = []
            for word in token_list:
                index = dictionary[word]
                index_list.append(index)
            dataset.append(index_list)
        return dictionary, reversed_dictionary, dataset

    def build_coocurrence(self):
        dictionary, reversed_dictionary, dataset = self.build_dataset()
        vocabulary_size = len(dictionary)
        coocurrence_matrix = np.zeros(shape=(vocabulary_size, vocabulary_size), dtype=float)

        for index_list in dataset:
            for i in range(len(index_list)):
                contexts_left = []
                contexts_right = []
                target = index_list[i]
                for j in range(1, self.window_size+1):
                    if i - j < 0:
                        contexts_left.append(0)
                    else:
                        context_left = index_list[i-j]
                        contexts_left.append(context_left)
                    if i + j > len(index_list)-1:
                        contexts_right.append(1)
                    else:
                        context_right = index_list[i+j]
                        contexts_right.append(context_right)
                for m in range(len(contexts_left)):
                    cleft = contexts_left[m]
                    cright = contexts_right[m]
                    distance = m+1
                    ratio = 1/distance
                    coocurrence_matrix[target][cleft] += ratio
                    coocurrence_matrix[target][cright] += ratio
        max_coocurrence = 0
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                coocurrence = coocurrence_matrix[i][j]
                if coocurrence == 0:
                    coocurrence_matrix[i][j] = self.min_coocurrence
                if coocurrence > max_coocurrence:
                    max_coocurrence = coocurrence
        return coocurrence_matrix, max_coocurrence


"""
gloVe 
"""


class GloVe:

    def __init__(self, train_data, val_data, coocurrence_matrix, reversed_dictionary, vocabulary_size, embedding_size,
                 window_size, max_coocurrence, scaling_factor, learning_rate, batch_size, num_epochs):
        self.train_data = train_data
        self.val_data = val_data
        self.coocurrence_matrix = coocurrence_matrix
        self.reversed_dictionary = reversed_dictionary
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.max_coocurrence = max_coocurrence
        self.scaling_factor = scaling_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.saver = tf.train.Saver()

    def generate_input_output_pair(self):
        pairs = []
        for token_list in self.train_data:
            for index in range(len(token_list)):
                target_word = self.train_data[index]
                context_words = []
                for num in range(1, self.window_size+1):
                    if index - num < 0:
                        context_words.append(0)
                    elif index + num > 0:
                        context_words.append(1)
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
            mini_batch = pairs[number*self.batch_size:number*self.batch_size+self.batch_size]
            targets = []
            contexts = []
            coocurrences = []
            for (target, context) in mini_batch:
                targets.append(target)
                contexts.append(context)
                coocurrences.append(self.coocurrence_matrix[target][context])
            batch_triple = (targets, contexts, coocurrences)
            batches.append(batch_triple)
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            batch_targets = tf.placeholder(tf.int32, shape=[self.batch_size, ]) #[batch_size, ]
            batch_contexts = tf.placeholder(tf.int32, shape=[self.batch_size, ]) #[batch_size, ]
            batch_coocurrences = tf.placeholder(tf.float32, shape=[self.batch_size, ]) #[batch_size, ]
            val_dataset = tf.constant(self.val_data, dtype=tf.int32)
            max_count = tf.constant(self.max_coocurrence, dtype=tf.int32)
            scaling_factor = tf.constant(self.scaling_factor, dtype=tf.float32)

        with tf.variable_scope('embeddings'):
            target_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            context_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            batch_targets_embeddings = tf.nn.embedding_lookup(target_embeddings, batch_targets) #[batch_size, embedding_size]
            batch_contexts_embeddings = tf.nn.embedding_lookup(context_embeddings, batch_contexts) #[batch_size, embedding_size]
            val_data_target_embeddings = tf.nn.embedding_lookup(target_embeddings, val_dataset) #[batch_size, embedding_size]
            val_data_context_embeddings = tf.nn.embedding_lookup(context_embeddings, val_dataset) #[batch_size, embedding_size]

        with tf.variable_scope('biases'):
            target_biases = tf.zeros([self.vocabulary_size], tf.float32)
            context_biaese = tf.zeros([self.vocabulary_size], tf.float32)
            batch_target_biases = tf.nn.embedding_lookup(target_biases, batch_targets) #[batch_size, 1]
            batch_context_biases = tf.nn.embedding_lookup(context_biaese, batch_contexts) #[batch_size, 1]

        with tf.name_scope('loss'):
            weighting = tf.minimum(tf.pow(tf.div(batch_coocurrences, max_count), scaling_factor), 1.0) #[batch_size, 1]
            dot_products = tf.reduce_sum(tf.matmul(batch_targets_embeddings, batch_contexts_embeddings, transpose_b=True), 1) #matmul[batch_size, batch_size], reduced_sum[batch_size, 1]
            log_coocurrences = tf.log(batch_coocurrences) #[batch_size,1]
            difference = tf.square(tf.add_n([dot_products, batch_target_biases, batch_context_biases,
                                            tf.negative(log_coocurrences)])) #[batch_size, 1]
            loss = tf.reduce_mean(tf.multiply(weighting, difference))

        with tf.name_scope('norm'):
            target_norm = tf.sqrt(tf.reduce_mean(tf.square(target_embeddings), 1, keep_dims=True))
            normed_target_embeddings = target_embeddings/target_norm
            context_norm = tf.sqrt(tf.reduce_mean(tf.square(context_embeddings), 1, keep_dims=True))
            normed_context_embeddings = context_embeddings/context_norm

        with tf.name_scope('similarity'):
            similariy_target = tf.matmul(val_data_target_embeddings, normed_target_embeddings, transpose_b=True)
            similariy_context = tf.matmul(val_data_context_embeddings, normed_context_embeddings, transpose_b=True)
        return batch_targets, batch_contexts, batch_coocurrences,  normed_target_embeddings, normed_context_embeddings, loss, similariy_target, similariy_context

    def train(self):
        batches = self.generate_batch()
        batch_targets, batch_contexts, batch_occurrences, normed_target_embeddings, normed_context_embeddings, loss, similarity_target, similarity_context = self.model()
        optimizer = tf.train.AdaGradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            for epoch in self.num_epochs:
                print('Number of epoch:', epoch)
                average_loss = 0.0
                for iteration, batch_data in enumerate(batches):
                    targets, contexts, occurrences = self.model()
                    feed_dict = {batch_targets: targets, batch_contexts: contexts, batch_occurrences: occurrences}
                    _, loss_val = session.run(optimizer, feed_dict)
                    average_loss += loss_val

                    if iteration % 1000 == 0:
                        if iteration > 0:
                            average_loss /= 1000
                    print('Loss at iteration ', iteration, ": ", average_loss)
                    average_loss = 0

                    if iteration % 5000 == 0:
                        sim_t = similarity_target.eval()
                        for i in range(len(self.val_data)):
                            top_10 = 10
                            nearest = (-sim_t[i, :]).argsort()[0:top_10+1]
                            for nw in nearest:
                                print(self.reversed_dictionary[nw], end=' ', flush=True)
                        sim_c = similarity_context.eval()
                        for i in range(len(self.val_data)):
                            top_10 = 10
                            nearest = (-sim_c[i, :]).argsort()[0:top_10+1]
                            for nw in nearest:
                                print(self.reversed_dictionary[nw], end=' ', flush=True)



