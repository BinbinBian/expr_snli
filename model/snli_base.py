import tensorflow as tf


class SnliLoader:
    def __init__(self, lstm_step=80, input_d=300, vocab_size=2196018, embedding=None):
        """
        Input   raw_premise (word ids)
                raw_hypothesis (word ids)
                lstm_step should be the max length of the input
        """
        self.raw_premise = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='premise')
        self.premise_length = tf.placeholder(shape=[None], dtype=tf.int32, name='premise_length')

        self.raw_hypothesis = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='hypothesis')
        self.hypothesis_length = tf.placeholder(shape=[None], dtype=tf.int32, name='hypothesis_length')

        self.label = tf.placeholder(shape=[None], dtype=tf.int32)

        if embedding is not None:
            self.input_embedding = tf.placeholder(dtype=tf.float32, shape=embedding.shape, name='word_embedding')
            self.embedding = tf.Variable(tf.zeros(embedding.shape, dtype=tf.float32))
        else:
            """
            If embedding is not provided, then use random number as embedding
            """
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, input_d], minval=-0.05, maxval=0.05))
        """
        This is the embedding operation. It will be invoked by loading embedding function in the actual model
        """
        self.load_embedding_op = self.embedding.assign(self.input_embedding)

        self.premise = tf.nn.embedding_lookup(self.embedding, self.raw_premise)
        self.hypothesis = tf.nn.embedding_lookup(self.embedding, self.raw_hypothesis)

    def load_embedding(self, sess, embedding):
        """
        :param sess: Session of the model, should be pass from the upper mode
        :param embedding: The embedding ndarray.
        :return:
        """
        sess.run(self.load_embedding_op, feed_dict={self.embedding: embedding})

    def feed_dict_builder(self, data):
        """
        :param data: The data should be a return tuple the snli_batchGenerator
        :return:
        """
        premise, p_len, hypothesis, h_len, label = data
        feed_dict = {
            self.raw_premise: premise,
            self.premise_length: p_len,
            self.raw_hypothesis: hypothesis,
            self.hypothesis_length: h_len,
            self.label: label
        }
        return feed_dict


class SnliSkipThoughtLoader:
    def __init__(self, sentence_d=2400):
        self.premise = tf.placeholder(dtype=tf.float32, shape=[None, sentence_d], name='premise')
        self.hypothesis = tf.placeholder(dtype=tf.float32, shape=[None, sentence_d], name='hypothesis')
        self.label = tf.placeholder(shape=[None], dtype=tf.int32)

    def feed_dict_builder(self, data):
        """
        :param data: The data should be a return tuple the snli_skipthought
        :return:
        """
        premise, hypothesis, label = data
        feed_dict = {
            self.premise: premise,
            self.hypothesis: hypothesis,
            self.label: label
        }
        return feed_dict
