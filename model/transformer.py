import tensorflow as tf
import numpy as np


class Transformer:
    def __init__(self, train_set, val_set, learning_rate, seq_length, n_class, n_hidden):
        self.train_set = train_set
        self.val_set = val_set
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self.n_class = n_class
        self.n_hidden = n_hidden

    def model(self):
        with tf.name_scope("input"):
            X = tf.placeholder(tf.int32, shape=[1, self.seq_length])
