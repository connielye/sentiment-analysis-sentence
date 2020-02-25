import tensorflow as tf
import numpy as np


class BiLSTMAttention:
    def __init__(self, train_data, val_data, n_hidden, n_class, seq_length, batch_size, learning_rate, epochs, word_embedding):
        self.train_data = train_data
        self.val_data = val_data
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.word_embedding = word_embedding

    def generate_batch(self):
        batches = []
        batch_number = len(self.train_data) // self.batch_size
        for number in range(batch_number):
            input_batches, target_batches = [], []
            batch = batches[number*self.batch_size: (number+1)*self.batch_size]
            for (input, target) in batch:
                input_batches.append(input)
                target_batches.append(target)
            batches.append((input_batches, target_batches))
        return batches

    def model(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, [None, self.batch_size, self.seq_length]) #[batch_size, seq_length]
            Y = tf.placeholder(tf.int32, [None, self.batch_size, 1])

        with tf.variable_scope('embedding'):
            embedding_weights = tf.get_variable("embedding_weights", shape=self.word_embedding.shape,
                                                initializer=tf.constant_initializer(self.word_embedding), trainable=False) #[vocab_size, embedding_size]
            embedding = tf.nn.embedding_lookup(embedding_weights, X) #[seq_length, embedding_size]
            embedding = tf.expand_dims(embedding, -1) #[1, seq_length, embedding_size]

        with tf.variable_scope('weights'):
            W = tf.Variable(tf.random_normal(self.n_hidden*2, self.n_class)) #[n_hidden*2, n_class]
            w_attn = tf.Variable(tf.random_normal(self.n_hidden*2, 1)) #[n_hidden*2, 1]

        with tf.variable_scope('bias'):
            B = tf.Variable(tf.random_normal(self.n_class))

        with tf.name_scope('bi-lstm'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedding, dtype=tf.float32)
            #outputs (outputs_fw, outputs_bw) each [batch_size, max_time, n_hidden]
            #output_states (output_state_fw, output_state_bw) each LSTMStateTuple(c, h) each [batch_size, n_hidden]

        with tf.name_scope('attention'):
            outputs = tf.concat(outputs, 2)# outputs [batch_size, max_time, n_hidden*2]
            M = tf.tanh(outputs) # M [batch, max_time, n_hidden*2]
            attn_weights = tf.matmul(M, w_attn) # [batch_size, max_time]
            soft_attn = tf.nn.softmax(attn_weights, 1)
            r = tf.matmul(tf.transpose(outputs, [0, 2, 1]), tf.expand_dims(soft_attn, 2)) #r [batch_size, n_hidden*2]
            out_attn = tf.tanh(r) # out_attn [batch_size, n_hidden*2]

        with tf.name_scope('loss'):
            ffn = tf.matmul(out_attn, W) + B
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y))

        with tf.name_scope('prediction'):
            prediction = tf.cast(tf.argmax(ffn, 1), tf.int32)

        return X, Y, cost, prediction

    def train(self):
        batches = self.generate_batch()
        X, Y, cost, prediction = self.model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        with tf.name_scope('session'):
            session = tf.Session()
            init = tf.global_variables_initializer()
            session.run(init)

            for epoch in range(self.epochs):
                for (input_batch, target_batch) in batches:
                    _, loss =session.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})

                    if (epoch+1) % 1000 == 0:
                        print('Epoch:', '%04d'%(epoch+1), 'cost =', '{:.6f}'.format(loss))