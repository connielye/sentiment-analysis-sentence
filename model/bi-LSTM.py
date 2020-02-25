import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.python.lib.io import file_io


class BiLSTM:
    def __init__(self, train_data, val_data, learning_rate, epochs, n_hidden, n_class, batch_size, embedding_size, sequence_length, vocab_size, saved_file, glove_embedding):
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.saved_file = saved_file
        self.glove_embedding = glove_embedding

    def generate_batch(self):
        batches = []
        batch_number = len(self.train_data)//self.batch_size
        for number in range(batch_number):
            batch = self.train_data[number*self.batch_size:(number+1)*self.batch_size]
            inputs_batch, targets_batch = [], []
            for (input, target) in batch:
                inputs_batch.append(input)
                targets_batch.append(target)
            batches.append((inputs_batch, targets_batch))
        return batches

    def model(self):
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length]) #[batch_size, sequence_length]
            Y = tf.placeholder(tf.int32, [self.batch_size, self.n_class]) #[batch_size, n_class]

        with tf.variable_scope('embedding'):
            glove_weights_initializer = tf.constant_initializer(self.glove_embedding)
            word_embedding = tf.get_variable(name="embedding_weights", shape=[self.vocab_size, self.embedding_size], initializer=glove_weights_initializer,
                                             trainable=False)
            embedding = tf.nn.embedding_lookup(word_embedding, X) #[batch_size, embedding_size]

        with tf.variable_scope('weights'):
            W = tf.Variable(tf.random_normal([self.n_hidden*2, self.n_class])) #[n_hidden*2, n_class]

        with tf.variable_scope('bias'):
            B = tf.Variable(tf.random_normal([self.n_class])) #[n_class]

        with tf.variable_scope('bilstm'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedding, dtype=tf.float32) #outputs(fw, bw) each [batch_size, max_time, n_hidden]
            outputs = tf.concat(outputs, 2) # outputs [batch_size, max_time, n_hidden*2]
            outputs = tf.transpose(outputs, [1, 0, 2]) # outputs [max_time, batch_size, n_hidden*2]
            outputs = outputs[-1] #outputs [batch_size, n_hidden*2]

        with tf.name_scope('cost'):
            ffn = tf.nn.xw_plus_b(outputs, W, B)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y))

        with tf.name_scope('predict'):
            predicts = tf.cast(tf.argmax(tf.nn.softmax(ffn), 1), tf.int32)
        return X, Y, cost, predicts

    def train(self):
        batches = self.generate_batch()
        X, Y, cost, predicts = self.model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            for epoch in range(self.epochs):

                total_loss = 0
                for (inputs, targets) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={X: inputs, Y: targets})
                    total_loss += loss

                if (epoch+1)%100 == 0:
                    average_loss = total_loss/len(batches)
                    print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

            a_count = 0
            for (input, target) in self.val_data:
                test = []
                test.append(input)
                predict = session.run([predicts], feed_dict={X:test})
                result = predict[0][0]
                if result == target:
                    a_count += 1
            accuracy = a_count/len(self.val_data)
            print(accuracy)

            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.saved_file)
            signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'input': tf.compat.v1.saved_model.utils.build_tensor_info(X)},
                                                                                         outputs={'output': tf.compat.v1.saved_model.utils.build_tensor_info(predicts)},
                                                                                         method_name="tensorflow/serving/predict")
            builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature}, strip_default_attrs=True)
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
    model = BiLSTM(train_data, val_data, 0.001, 1000, 128, 3, 1, 300, 128, vocab_size, saved_file, embeddings)
    model.train()

train_file = "/home/bingxin/Documents/trainingdata/train.csv"
glove_path = "/home/bingxin/Downloads/glove.twitter.27B.100d.txt"
job_dir = "/home/bingxin/Documents/tmp/SENT_LSTM_1"
train_model(train_file, glove_path, job_dir)
