import tensorflow as tf
import numpy as np


"""
Implement algorithms to generate word embedding adding sentiment weights. 
DLJT1 (the objective function of incorporating document-level sentiment distribution of target word) and 
DLJC1 (the objective function of incorporating document-level sentiment distribution of context word) are NOT implemented
as the experiment results of the two algorithms are not impressive from the paper.
Li, Yang, et al. "Learning word representations for sentiment analysis." Cognitive Computation 9.6 (2017): 843-851.
"""

"""
build datasets with results from prior_sentiment_knowledge file from sentiment word level and document level
"""


class BuildDataset:

    def __init__(self, data, dictionary, reversed_dictionary, coocurrence_matrix, window_size, dl_senti_info, dl_senti_ratio):
        self.data = data
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary
        self.coocurrence_matrix = coocurrence_matrix
        self.window_size = window_size
        self.dl_senti_info = dl_senti_info
        self.dl_senti_ratio = dl_senti_ratio

    def build_wl_dataset(self, wl_dictionary, wl_reversed_dictionary, wl_senti_ratio):
        dataset = []
        for index_list in self.data:
            for i in range(len(index_list)):
                target = index_list[i]
                target_word = self.reversed_dictionary[target]
                target_senti_info = self.dl_senti_info[target]
                target_senti_ratio = 1
                if target_word in wl_reversed_dictionary.values():
                    target_senti_index = wl_dictionary[target_word]
                    target_senti_ratio = wl_senti_ratio[target_senti_index]
                contexts = []
                for j in range(self.window_size):
                    if i - j < 0:
                        contexts.append(0)
                    else:
                        contexts.append(index_list[i-j])
                    if i + j > len(index_list):
                        contexts.append(1)
                    else:
                        contexts.append(index_list[i+j])
                for context in contexts:
                    context_senti_info = self.dl_senti_info[context]
                    context_word = self.reversed_dictionary[context]
                    context_senti_ratio = 1
                    if context_word in wl_reversed_dictionary.values():
                        context_senti_index = wl_dictionary[context_word]
                        context_senti_ratio = wl_senti_ratio[context_senti_index]
                    coocurrence = self.coocurrence_matrix[target][context]
                    item = (target, context, coocurrence, target_senti_info, target_senti_ratio, context_senti_info,
                            context_senti_ratio)
                    dataset.append(item)
        return dataset

    def build_dl_dataset(self):
        dataset = []
        coocurrences = []
        context_infos = []
        context_ratios = []
        target_infos = []
        target_ratios = []
        for index_list in self.data:
            for i in range(len(index_list)):
                target = index_list[i]
                target_senti_info = self.dl_senti_info[target]
                target_infos.append(target_senti_info)
                target_senti_ratio = self.dl_senti_ratio[target]
                target_ratios.append(target_senti_ratio)
                contexts = []
                for j in range(self.window_size):
                    if i - j < 0:
                        contexts.append(0)
                    else:
                        contexts.append(index_list[i-j])
                    if i + j > 0:
                        contexts.append(1)
                    else:
                        contexts.append(index_list[i+j])
                for context in contexts:
                    context_senti_info = self.dl_senti_info[context]
                    context_infos.append(context_senti_info)
                    context_senti_ratio = self.dl_senti_ratio[context]
                    context_ratios.append(context_senti_ratio)
                    coocurrence = self.coocurrence_matrix[target][context]
                    coocurrences.append(coocurrence)
                    item = (target, context, coocurrence, target_senti_info, target_senti_ratio, context_senti_info,
                            context_senti_ratio)
                    dataset.append(item)
        return dataset


"""
WLJT: the objective function of incorporating word-level sentiment ratio of target word
DLJT2: the objective function of incorporating document-level sentiment ratio of target word
"""


class WLJT_DLJT2:

    def __init__(self, dataset, val_data, embedding_size, vocabulary_size, max_coocurrence, scaling_factor, batch_size,
                 learning_rate, num_epochs, reversed_dictionary):
        self.dataset = dataset
        self.val_data = val_data
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.max_coocurrence = max_coocurrence
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.reversed_dictionary = reversed_dictionary
        self.saver = tf.train.Saver()

    def generate_batch(self):
        batches = []
        batch_number = len(self.dataset) // self.batch_size
        for i in range(batch_number):
            item = self.dataset[i*self.batch_size:i*self.batch_size+self.batch_size]
            targets = []
            contexts = []
            coocurrences = []
            target_senti_ratios = []
            for target, context, coocurrence, target_senti_info, target_senti_ratio, context_senti_info, \
                context_senti_ratio in item:
                targets.append(np.asarray(target))
                contexts.append(np.asarray(context))
                coocurrences.append(np.asarray(coocurrence))
                target_senti_ratios.append(np.asarray(target_senti_ratio))
            batch = (targets, contexts, coocurrences, target_senti_ratios)
            batches.append(batch)
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            batch_targets = tf.placeholder(tf.int64, [self.batch_size,])
            batch_contexts = tf.placeholder(tf.int64, [self.batch_size,])
            batch_coocurrence = tf.placeholder(tf.float64, [self.batch_size,])
            batch_target_senti_ratio = tf.placeholder(tf.float64, [self.batch_size,])
            max_count = tf.constant(self.max_coocurrence)
            val_data = tf.constant(self.val_data)

        with tf.variable_scope('embeddings'):
            target_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            context_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size],
                                                               -1.0, 1.0))
            batch_target_embeddings = tf.nn.embedding_lookup (target_embeddings, batch_targets)
            batch_context_embeddings = tf.nn.embedding_lookup(context_embeddings, batch_contexts)
            val_data_target_embeddings = tf.nn.embedding_lookup(target_embeddings, val_data)
            val_data_context_embeddings = tf.nn.embedding_lookup(context_embeddings, val_data)

        with tf.variable_scope('senti-weights'):
            senti_weights = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            target_senti_weights = tf.nn.embedding_lookup(senti_weights, batch_targets)

        with tf.variable_scope('biases'):
            target_biases = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            context_biases = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            batch_target_biases = tf.nn.embedding_lookup(target_biases, batch_targets)
            batch_context_biases = tf.nn.embedding_lookup(context_biases, batch_contexts)

        with tf.name_scope('loss'):
            weighting = tf.minimum(tf.pow(tf.div(batch_coocurrence, max_count), self.scaling_factor), 1.0)
            dot_products = tf.reduce_sum(tf.matmul(batch_target_embeddings, batch_context_embeddings, transpose_b=True), 1)
            log = tf.log(tf.multiply(batch_coocurrence, batch_target_senti_ratio))
            difference = tf.square(tf.add_n([tf.multiply(dot_products, target_senti_weights), batch_target_biases, batch_context_biases,
                                  tf.negative(log)]))
            loss = tf.reduce_mean(tf.multiply(weighting, difference))

        with tf.name_scope('norm'):
            target_norm = tf.sqrt(tf.reduce_mean(tf.square(batch_target_embeddings), 1, keep_dims=True))
            normed_batch_target_embeddings = batch_target_embeddings / target_norm
            context_norm = tf.sqrt(tf.reduce_mean(tf.square(batch_context_embeddings), 1, keep_dims=True))
            normed_batch_context_embeddings = batch_context_embeddings / context_norm

        with tf.name_scope('similarity'):
            similarity_target = tf.matmul(val_data_target_embeddings, normed_batch_target_embeddings, transpose_b=True)
            similarity_context = tf.matmul(val_data_context_embeddings, normed_batch_context_embeddings,
                                           transpose_b=True)
        return batch_targets, batch_contexts, batch_coocurrence, batch_target_senti_ratio, \
               normed_batch_target_embeddings, normed_batch_context_embeddings, loss, similarity_target, \
               similarity_context

    def train(self):
        batches = self.generate_batch()
        batch_targets, batch_contexts, batch_coocurrence, batch_target_senti_info, batch_target_senti_ratio, \
        normed_batch_target_embeddings, normed_batch_context_embeddings, loss, similarity_target, \
        similarity_context = self.model()
        optimizer = tf.train.AdamGradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session as session:
            session.run(init)
            self.saver.save(session, "test-model", global_step=1000)

            for epoch in range(self.num_epochs):
                print('Number of Epoch:', epoch)

                average_loss = 0.0
                for iteration, batch_data in enumerate(batches):
                    targets, contexts, coocurrences, target_senti_info_list, target_senti_ratios = batch_data
                    feed_dict = {batch_targets: targets, batch_contexts: contexts, batch_coocurrence: coocurrences,
                                 batch_target_senti_info: target_senti_info_list,
                                 batch_target_senti_ratio: target_senti_ratios}
                    _, val_loss = session.run([optimizer, loss], feed_dict)
                    average_loss += val_loss

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


"""
WLJC: the objective function incorporating word-level sentiment ratio of context word
DLJC2: the objective function incorporating document-level sentiment ratio of context word
"""


class WLJC_DLJC2:

    def __init__(self, dataset, embedding_size, vocabulary_size, max_coocurrence, scaling_factor, batch_size,
                 learning_rate, num_epochs, reversed_dictionary):
        self.dataset = dataset
        #self.val_data = val_data
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.max_coocurrence = max_coocurrence
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.reversed_dictionary = reversed_dictionary

    def generate_batch(self):
        batches = []
        batch_number = len(self.dataset) // self.batch_size
        for i in range(batch_number):
            item = self.dataset[i*self.batch_size:i*self.batch_size+self.batch_size]
            targets = []
            contexts = []
            coocurrences = []
            context_senti_info_list = []
            context_senti_ratios = []
            for target, context, coocurrence, target_senti_info, target_senti_ratio, context_senti_info, \
                context_senti_ratio in item:
                targets.append(np.asarray(target))
                contexts.append(np.asarray(context))
                coocurrences.append(np.asarray(coocurrence))
                context_senti_ratios.append(np.asarray(context_senti_ratio))
            batch = (targets, contexts, coocurrences, context_senti_info_list, context_senti_ratios)
            batches.append(batch)
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            batch_targets = tf.placeholder(tf.int32, [self.batch_size, ]) #[batch_size, 1]
            batch_contexts = tf.placeholder(tf.int32, [self.batch_size,]) #[batch_size, 1]
            batch_coocurrence = tf.placeholder(tf.float32, [self.batch_size,]) #[batch_size, 1]
            batch_context_senti_ratio = tf.placeholder(tf.float32, [self.batch_size,]) #[batch_size, 1]
            max_count = tf.constant(self.max_coocurrence, dtype=tf.float32)

        with tf.variable_scope('embeddings'):
            target_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            context_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            batch_target_embeddings = tf.nn.embedding_lookup(target_embeddings, batch_targets) #[batch_size, embedding_size]
            batch_context_embeddings = tf.nn.embedding_lookup(context_embeddings, batch_contexts) #[batch_size, embedding_size]
            #val_data_target_embeddings = tf.nn.embedding_lookup(target_embeddings, val_data)
            #val_data_context_embeddings = tf.nn.embedding_lookup(context_embeddings, val_data)

        with tf.variable_scope('senti-weights'):
            senti_weights = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            context_senti_weights = tf.nn.embedding_lookup(senti_weights, batch_contexts) #[batch_size, 1]

        with tf.variable_scope('biases'):
            target_biases = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            context_biases = tf.Variable(tf.zeros([self.vocabulary_size], tf.float32))
            batch_target_biases = tf.nn.embedding_lookup(target_biases, batch_targets) #[batch_size, 1]
            batch_context_biases = tf.nn.embedding_lookup(context_biases, batch_contexts) #[batch_size, 1]

        with tf.name_scope('loss'):
            weighting = tf.minimum(tf.pow(tf.div(batch_coocurrence, max_count), self.scaling_factor), 1.0) #[batch_size, 1]
            dot_products = tf.reduce_sum(tf.matmul(batch_target_embeddings, batch_context_embeddings, transpose_b=True), 1) #[batch_size, 1]
            log = tf.log(tf.multiply(batch_coocurrence, batch_context_senti_ratio)) #[batch_size, 1]
            difference =tf.square(tf.add_n([tf.multiply(dot_products, context_senti_weights), batch_target_biases, batch_context_biases, tf.negative(log)])) #[batch_size, 1]
            loss = tf.reduce_mean(tf.multiply(weighting, difference))

        with tf.name_scope('norm'):
            target_norm = tf.sqrt(tf.reduce_mean(tf.square(batch_target_embeddings), 1, keep_dims=True))
            normed_batch_target_embeddings = batch_target_embeddings / target_norm
            context_norm = tf.sqrt(tf.reduce_mean(tf.square(batch_context_embeddings), 1, keep_dims=True))
            normed_batch_context_embeddings = batch_context_embeddings / context_norm

        with tf.name_scope('similarity'):
            """
            similarity_target = tf.matmul(val_data_target_embeddings, normed_batch_target_embeddings, transpose_b=True)
            similarity_context = tf.matmul(val_data_context_embeddings, normed_batch_context_embeddings,
                                           #transpose_b=True)
            """
        return batch_targets, batch_contexts, batch_coocurrence, batch_context_senti_ratio, \
               normed_batch_target_embeddings, normed_batch_context_embeddings, loss, #similarity_target, \
               #similarity_context

    def train(self):
        batches = self.generate_batch()
        batch_targets, batch_contexts, batch_coocurrence, batch_context_senti_ratio, \
        normed_batch_target_embeddings, normed_batch_context_embeddings, loss = self.model()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        #builder = tf.saved_model.builder.SavedModelBuilder("")
        with tf.Session() as session:
            session.run(init)

            tf.train.Saver().save(session, '/home/bingxin/Documents/test/temp/test_model2')
            for epoch in range(self.num_epochs):
                print('Number of Epoch:', epoch)

                average_loss = 0.0
                for iteration, batch_data in enumerate(batches):
                    targets, contexts, coocurrences, context_senti_info_list, context_senti_ratios = batch_data
                    feed_dict = {batch_targets: targets, batch_contexts: contexts, batch_coocurrence: coocurrences,
                                 batch_context_senti_ratio: context_senti_ratios}
                    _, val_loss = session.run([optimizer, loss], feed_dict)
                    #average_loss += val_loss

                    if iteration % 1000 == 0:
                        """
                        if iteration > 0:
                            average_loss /= 1000
                        print('Loss at iteration ', iteration, ": ", average_loss)
                        average_loss = 0
                        """
                        print('cost at iteration', iteration, 'cost =', '{:.6f}'.format(val_loss))
        #builder.save()
"""
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
"""
