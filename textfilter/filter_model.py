import tensorflow as tf
import numpy as np
import re



class Filter:
    def __init__(self, train_set, val_set, glove_weights, learning_rate, epochs, n_class, embedding_size, n_hidden, sequence_length, batch_size, vocab_size, saved_file):
        self.train_set = train_set
        self.val_set = val_set
        self.glove_weights = glove_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_class = n_class
        self.embedding_size = embedding_size
        self.n_hidden = n_hidden
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.saved_file = saved_file


    def generate_batch(self):
        batches = []
        number_batch = len(self.train_set)//self.batch_size
        for number in range(number_batch):
            batch = self.train_set[number*self.batch_size: (number+1)*self.batch_size]
            input_batch, target_batch = [], []
            for (input, target) in batch:
                input_batch.append(input)
                target_batch.append(target)
            batches.append((input_batch, target_batch))
        return batches

    def model(self):
        with tf.name_scope("inputs"):
            X = tf.placeholder(dtype=tf.int32, shape=[1, self.sequence_length], name="input")
            Y = tf.placeholder(dtype=tf.int32, shape=[1, self.n_class])

        with tf.variable_scope("embedding"):
            glove_weights_initializer = tf.constant_initializer(self.glove_weights)
            glove_embeddings = tf.get_variable(name="glove_embedding", shape=[self.vocab_size, self.embedding_size], initializer=glove_weights_initializer, trainable=False)
            embeddings = tf.nn.embedding_lookup(glove_embeddings, X)
            embeddings = tf.expand_dims(embeddings, -1)

        with tf.variable_scope('weights'):
            W = tf.Variable(tf.random_normal(shape=[self.n_hidden*2, self.n_class], dtype=tf.float32))

        with tf.variable_scope("bias"):
            b = tf.Variable(tf.random_normal(shape=[self.n_class], dtype=tf.float32))

        with tf.variable_scope("biLSTM"):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, dtype=tf.float32)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embeddings, dtype=tf.float32) # [batch_size, max_time, n_hidden]
            outputs = tf.concat(outputs, 2) #[batch_size, max_time, n_hidden*2]
            outputs = tf.transpose(outputs, [1, 0, 2]) #[max_time, batch_size, n_hidden*2]
            outputs = outputs[-1] #[batch_size, n_hidden*2]

        with tf.variable_scope("cost"):
            ffn = tf.nn.xw_plus_b(outputs, W, b)
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y)

        with tf.variable_scope("prediction"):
            predict = tf.cast(tf.argmax(tf.nn.softmax(cost), 1), dtype=tf.int32, name="output")

        return X, Y, cost, predict



    def train(self):

        batches = self.generate_batch()
        X, Y, cost, predicts = self.model()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)
        init = tf.compat.v1.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
            session.run(init)

            for epoch in range(self.epochs):
                total_cost = 0

                for (input_batch, target_batch) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
                    total_cost += loss

                if (epoch+1)%100 == 0:
                    average_cost = total_cost/len(batches)
                    print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_cost))


            accu_count = 0
            for(input, target) in self.val_set:
                val_input = []
                val_input.append(input)
                prediction = session.run([predicts], feed_dict={X:val_input})
                if prediction[0][0] == target:
                    accu_count += 1
            accuracy = float(accu_count)/float(len(self.val_set))
            print("Accuracy:", "{:.6f}".format(accuracy))

            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.saved_file)
            signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'input': tf.compat.v1.saved_model.utils.build_tensor_info(X)},
                                                                                         outputs={'output': tf.compat.v1.saved_model.utils.build_tensor_info(predicts)},
                                                                                         method_name="tensorflow/serving/predict")
            builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature},
                                                 strip_default_attrs=True)
            builder.save()


def train_model(data_file, glove_file, export_dir):

    with open(data_file, 'r') as df:
        data = df.readlines()
    sentences =[]
    labels = []
    for line in data:
        parts = line.rstrip().split('\t')
        sentences.append(parts[0])
        labels.append(int(parts[1]))

    with open(glove_file, 'r') as gf:
        glove_data = gf.readlines()

    print("building vectors and dictionary...")

    vocab = []
    vectors = []
    for row in glove_data:
        line = row.rstrip().split(' ')
        vocab.append(line[0])
        vectors.append(np.asarray(line[1:]))

    vectors.insert(0, np.random.randn(300))
    vectors.append(np.random.randn(300))
    embeddings = np.asarray(vectors)
    vocab.insert(0, '<PAD>')
    vocab.append('<UNK>')

    vocab_size = len(vocab)
    dictionary = {w:i for i, w in enumerate(vocab)}

    print("building input dataset ...")

    inputs = []
    for sentence in sentences:
        tokens = re.split("\W+", sentence.rstrip().lower())
        for i in range(len(tokens)):
            token = tokens[i]
            if token == "":
                tokens.remove(token)
            elif token not in vocab:
                tokens[i] = "<UNK>"
        for i in range(len(tokens), 128):
            tokens.append("<PAD>")
        input_sentence = np.asarray([dictionary[token] for token in tokens])
        inputs.append(input_sentence)

    targets = []
    for lable in labels:
        target = [np.eye(2)[lable]]
        targets.append(target)

    dataset = []
    for i in range(len(inputs)):
        dataset.append((inputs[i], targets[i]))

    train_set = dataset[: int(len(dataset)*0.8)]
    val_set = dataset[int(len(dataset)*0.2):]

    filter = Filter(train_set, val_set, embeddings, 0.001, 1000, 2, 300, 128, 128, 1, vocab_size, export_dir)
    filter.train()






