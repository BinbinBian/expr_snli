from model.snli_base import SnliSkipThoughtLoader
from model.base import record_info, nn_layer
import tensorflow as tf
import numpy as np


class BasicSkipThought:
    def __init__(self, sentence_d=4800, softmax_keeprate=0.8, learning_rate=0.001, num_class=3, **kwargs):
        self.model_info = record_info(SkipThoughtDimension=sentence_d,
                                      Softmax_Keep_Rate=softmax_keeprate,
                                      kwargs=kwargs)

        self.input_loader = SnliSkipThoughtLoader(sentence_d=sentence_d)
        self.sentence_embedding_output = tf.concat(1, [self.input_loader.premise, self.input_loader.hypothesis,
                                                       tf.abs(
                                                           self.input_loader.premise - self.input_loader.hypothesis),
                                                       tf.mul(self.input_loader.premise, self.input_loader.hypothesis)])

        self.output = nn_layer(tf.nn.dropout(self.sentence_embedding_output, keep_prob=softmax_keeprate),
                               shape=[sentence_d * 4, num_class],
                               name='affine-softmax',
                               w_init=tf.contrib.layers.xavier_initializer(uniform=False),
                               b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                               act=None)

        self.softmax_output = tf.nn.softmax(self.output)
        self.prediction = tf.argmax(self.softmax_output, dimension=1)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.input_loader.label)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()

    def train(self, feed_dict):
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, feed_dict):
        y_pred = feed_dict[self.input_loader.label]
        out_pred, out_cost = self.sess.run((self.prediction, self.cost), feed_dict=feed_dict)
        accuracy = np.sum(y_pred == out_pred) / len(y_pred)
        return accuracy, (np.sum(out_cost) / len(out_cost))

    def setup(self, **info):
        self.sess.run(self.init_op)
        newinfo = record_info(Info=info)
        """
        Update the information about the model after load embedding.
        """
        for k, v in newinfo.items():
            self.model_info[k] = v

    def close(self):
        self.sess.close()
