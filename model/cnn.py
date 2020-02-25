import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.python.lib.io import file_io
import argparse


class CNN:
    def __init__(self, train_data, val_data, learning_rate, epochs, n_class, batch_size, embedding_size, sequence_length, filter_sizes, num_filters, vocab_size, saved_file):
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_class = n_class
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.saved_file = saved_file

    def generate_batch(self):
        batches = []
        number_batch = len(self.train_data)//self.batch_size
        for number in range(number_batch):
            batch = self.train_data[number*self.batch_size: (number+1)*self.batch_size]
            input_batch, target_batch = [], []
            for (input, target) in batch:
                input_batch.append(input)
                target_batch.append(target)
            batches.append((input_batch, target_batch))
        return batches

    def model(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='input') #[batch_size, max_sentence_length]
            Y = tf.placeholder(tf.int32, [self.batch_size, self.n_class]) #[batch_size, n_class]

        with tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -0.1, 0.1))
            embedding = tf.nn.embedding_lookup(W, X) #[batch_size, sequence_length, embedding_size]
            embedding = tf.expand_dims(embedding, -1)
            b = tf.Variable(tf.constant(0.1, shape=[self.n_class])) #[batch_size, n_class]

        with tf.name_scope('convolution'):
            outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W_cov = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b_cov = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))

                conv = tf.nn.conv2d(embedding, W_cov, strides=[1,1,1,1], padding='VALID')
                h_act = tf.nn.relu(tf.nn.bias_add(conv, b_cov))

                max_pooled = tf.nn.max_pool(h_act, ksize=[1, self.sequence_length-filter_size+1, 1, 1], strides=[1,1,1,1], padding='VALID')
                outputs.append(max_pooled)
            t_num_filters = self.num_filters*len(self.filter_sizes)
            h_concat = tf.concat(outputs, self.num_filters)
            h_concat_flat = tf.reshape(h_concat, [-1, t_num_filters])

        with tf.name_scope('cost'):
            weight = tf.get_variable('W', shape=[t_num_filters, self.num_filters], initializer=tf.contrib.layers.xavier_initializer())
            ffn = tf.nn.xw_plus_b(h_concat_flat, weight, b)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y))

        with tf.name_scope('predict'):
            predicts = tf.cast(tf.argmax(tf.nn.softmax(ffn), 1), tf.int32, name='output')

        return X, Y, cost, predicts

    def train(self):
        #with tf.device("'/device:GPU:0"):
            batches = self.generate_batch()
            X, Y, cost, predicts = self.model()
            optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)

            init = tf.compat.v1.global_variables_initializer()


            with tf.Session() as session:
                session.run(init)

                for epoch in range(self.epochs):

                    total_loss = 0
                    for (inputs, targets) in batches:
                        _, loss = session.run([optimizer, cost], feed_dict={X: inputs, Y: targets})
                        total_loss += loss

                    if (epoch+1)%100 == 0:
                        average_loss = total_loss/len(batches)
                        print ("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                a_count = 0
                for (input, target) in self.val_data:
                    test = []
                    test.append(input)
                    predict = session.run([predicts], feed_dict={X:test})
                    result = predict[0][0]
                    if result == target:
                        a_count += 1
                accuracy = float(a_count)/float(len(self.val_data))
                print("Accuracy:", accuracy)


                signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'input': tf.compat.v1.saved_model.utils.build_tensor_info(X)},
                                                                       outputs={'output': tf.compat.v1.saved_model.utils.build_tensor_info(predicts)},
                                                                                                method_name="tensorflow/serving/predict")
                builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.saved_file)
                builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
                builder.save(True)




def train_model_cnn(train_file, dictionary_file, job_dir,):

    print("loading the training file ...")
    with file_io.FileIO(train_file, 'r') as input_file:
        input_data = pd.read_csv(input_file, sep=';', engine='python', header=None, names=['Index', 'Text', 'Label'])

    print("loading the dictionary file ...")
    dictionary={}
    reversed_dictionary = {}
    with file_io.FileIO(dictionary_file, 'r') as df:
        for line in df:
            ts = line.rstrip().split('\t')
            dictionary[ts[1]] = int(ts[0])
            reversed_dictionary[int(ts[0])] = ts[1]

    df = pd.DataFrame(input_data)

    text = pd.Series(df['Text'])
    label = pd.Series(df['Label'])

    token_list, tokens = [], []
    for row in text:
        row_string = str(row)
        token = re.split("\W+", row_string)
        for t in token:
            if t == '':
                token.remove(t)
        token_list.append(token)
        tokens.extend(token)


    vocab_size = len(dictionary)

    keys = dictionary.keys()

    inputs = []
    for list in token_list:
        if len(list) < 128:
            for i in range(len(list), 128):
                list.append("PAD")
        for i, t in enumerate(list):
            if t not in keys:
                list[i] = "UNK"
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

    print("training the model ...")
    saved_file = job_dir + "/saved_model"

    model = CNN(train_set, val_set, 0.01, 1000, 3, 1, 300, 128, [1,2,3], 3, vocab_size, saved_file)
    model.train()

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', help='GCS or local paths to training data', required=True)
    parser.add_argument('--dictionary-file', help='GCS or local paths to dictionary', required=True)
    parser.add_argument('--job-dir', help='GCS location to save model', required=True)

    args = parser.parse_args()
    arguments = args.__dict__

    train_model_cnn(**arguments)
"""
train_file = "/home/bingxin/Documents/trainingdata/test.csv"
dictionary_file = "/home/bingxin/Documents/trainingdata/dictionary.txt"
job_dir = "/home/bingxin/Documents/tmp/testModel"

train_model_cnn(train_file, dictionary_file, job_dir)