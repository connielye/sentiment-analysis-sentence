import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.python.lib.io import file_io
import argparse

class GloveCNN:
    def __init__(self, train_data, val_data, learning_rate, epochs, n_class, batch_size, embedding_size, sequence_length, filter_sizes, num_filters, vocab_size, glove_embeddings, saved_file):
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_class = n_class
        self.batch_size = batch_size
        self. embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.glove_embeddings = glove_embeddings
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
        with tf.name_scope("input"):
            X = tf.placeholder(dtype=tf.int32, shape=[1, self.sequence_length], name="input") #[1, 128]
            Y = tf.placeholder(dtype=tf.int32, shape=[1, self.n_class]) #[1, 3]

        with tf.name_scope("embedding"):
            glove_weights_initializer = tf.constant_initializer(self.glove_embeddings)
            embedding_weights = tf.get_variable(name="embedding_weights", shape=[self.vocab_size, self.embedding_size],
                                                initializer=glove_weights_initializer, trainable=False) #[vocab_size, emb_size]
            embedding = tf.nn.embedding_lookup(embedding_weights, X) #[128, emb_size]
            embeddings = tf.expand_dims(embedding, -1) #[1, 128, emb_size]

        with tf.name_scope("convolution"):
            outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters] #[filter_size, emb_size, 1, 3]
                w_con = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1))
                b_con = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))

                conv = tf.nn.conv2d(embeddings, w_con, strides=[1,1,1,1], padding='VALID')
                h_act = tf.nn.relu(tf.nn.bias_add(conv, b_con))

                max_pool = tf.nn.max_pool(h_act, ksize=[1, self.sequence_length-filter_size+1, 1, 1], strides=[1,1,1,1], padding='VALID')
                outputs.append(max_pool)
            t_num_filters = self.num_filters*len(self.filter_sizes)
            h_concat = tf.concat(outputs, self.num_filters)
            h_concat_flat = tf.reshape(h_concat, shape=[-1, t_num_filters])

        with tf.name_scope("cost"):
            W = tf.Variable(tf.random_uniform(shape=[t_num_filters,self.num_filters], minval=-0.1, maxval=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.n_class]))
            ffn = tf.nn.xw_plus_b(h_concat_flat, W, b)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y))

        with tf.name_scope('predict'):
            predicts = tf.cast(tf.argmax(tf.nn.softmax(ffn), 1), dtype=tf.int32, name='output')

        return X, Y, cost, predicts

    def train(self):
        #with tf.device('/device:GPU:0'):
            batches = self.generate_batch()
            X, Y, cost, predicts = self.model()
            opimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)
            init = tf.compat.v1.global_variables_initializer()

            with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
                session.run(init)

                for epoch in range(self.epochs):
                    total_loss = 0
                    for (input, target) in batches:
                        _, loss = session.run([opimizer, cost], feed_dict={X:input, Y:target})
                        total_loss += loss

                    if (epoch+1)%100 == 0:
                        average_loss = total_loss/len(self.train_data)
                        print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                a_count = 0
                for (input, target) in self.val_data:
                    test =[]
                    test.append(input)
                    predict = session.run([predicts], feed_dict={X:test})
                    if predict[0][0] == target:
                        a_count+=1
                accuracy = float(a_count)/float(len(self.val_data))
                print("Accuracy:", accuracy)

                builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.saved_file)
                signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'input': tf.compat.v1.saved_model.utils.build_tensor_info(X)},
                                                                                         outputs={'output': tf.compat.v1.saved_model.utils.build_tensor_info(predicts)},
                                                                                             method_name="tensorflow/serving/predict")
                builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature}, strip_default_attrs=True)
                builder.save()


def train_model(train_file, glove_path, job_dir, **args):

    with file_io.FileIO(train_file, 'r') as t_file:
        data = pd.read_csv(t_file, sep=';', header=None, names=['Index', 'Text', 'Sentiment'], engine='python')

    with file_io.FileIO(glove_path, 'r') as glove_file:
        glove_data = glove_file.readlines()

    print("building vectors and dictionary...")
    vocab = []
    vectors = []
    for row in glove_data:
        line = row.rstrip().split(' ')
        vocab.append(line[0])
        vectors.append(np.asarray(line[1:]))

    vectors.insert(0, np.random.randn(100))
    vectors.append(np.random.randn(100))
    embeddings = np.asarray(vectors)
    vocab.insert(0, '<PAD>')
    vocab.append('<UNK>')

    vocab_size = len(vocab)
    dictionary = {w:i for i, w in enumerate(vocab)}

    df = pd.DataFrame(data)
    text = pd.Series(df['Text'])
    sentiment = pd.Series(df['Sentiment'])

    print("building dataset...")
    inputs = []
    for row in text:
        tokens = re.split('\W+', str(row).rstrip().lower())
        for token in tokens:
            if token =='':
                tokens.remove(token)
        if len(tokens)<128:
            for i in range(len(tokens), 128):
                tokens.append('<PAD>')
        token_list = []
        for token in tokens:
            if token not in vocab:
                token_list.append(dictionary["<UNK>"])
            else:
                token_list.append(dictionary[token])
        inputs.append(np.asarray(token_list))

    targets = []
    for row in sentiment:
        target = int(row)
        targets.append(np.eye(3)[target])

    input_data = []
    for i in range(len(inputs)):
        input_data.append((inputs[i], targets[i]))

    np.random.shuffle(input_data)

    train_data = input_data[:int(len(input_data)*0.8)]
    val_set = input_data[int(len(input_data)*0.8):]

    val_data = []
    for (input, target) in val_set:
        val_target = list(target).index(1)
        val_data.append((input, val_target))

    print("start training ...")
    saved_file = job_dir+"/saved_model"
    glove_cnn = GloveCNN(train_data, val_data, 0.001, 3000, 3, 1, 100, 128, [1,2,3], 3, vocab_size, embeddings, saved_file)
    glove_cnn.train()

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', help='GCS or local paths to training data', required=True)
    parser.add_argument('--glove-path', help='GCS or local paths to glove vectors', required=True)
    parser.add_argument('--job-dir', help='GCS or local paths to job dir', required=True)

    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)

"""

train_file = "/home/bingxin/Documents/trainingdata/train.csv"
glove_path = "/home/bingxin/Downloads/glove.twitter.27B.100d.txt"
job_dir = "/home/bingxin/Documents/tmp/SENT_serv"
train_model(train_file, glove_path, job_dir)