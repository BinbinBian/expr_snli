from model.base import BasicSeqModel, record_info, nn_layer
from model.snli_base import SnliLoader
import tensorflow as tf
import numpy as np


class SnliBasicLSTM:
    def __init__(self, lstm_step=80, input_d=300, vocab_size=2196018, hidden_d=100, num_class=3, learning_rate=0.001,
                 softmax_keeprate=0.75, lstm_input_keep_rate=1.0, lstm_output_keep_rate=0.90, embedding=None,
                 **kwargs):
        self.model_info = record_info(LSTM_Step=lstm_step,
                                      Word_Dimension=input_d,
                                      Vocabluary_Size=vocab_size,
                                      LSTM_Hidden_Dimension=hidden_d,
                                      Number_Class=num_class,
                                      SoftMax_Keep_Rate=softmax_keeprate,
                                      LSTM_Input_Keep_Rate=lstm_input_keep_rate,
                                      LSTM_Output_Keep_Rate=lstm_output_keep_rate,
                                      kwargs=kwargs)

        self.input_loader = SnliLoader(lstm_step, input_d, vocab_size, embedding)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_d, state_is_tuple=True)
        basic_seq_lstm_premise = BasicSeqModel(input_=self.input_loader.premise,
                                               length_=self.input_loader.premise_length,
                                               hidden_state_d=hidden_d,
                                               name='premise-lstm', cell=[lstm_cell],
                                               input_keep_rate=lstm_input_keep_rate,
                                               output_keep_rate=lstm_output_keep_rate)
        basic_seq_lstm_hypothesis = BasicSeqModel(input_=self.input_loader.hypothesis,
                                                  length_=self.input_loader.hypothesis_length,
                                                  hidden_state_d=hidden_d,
                                                  name='hypothesis-lstm', cell=[lstm_cell],
                                                  input_keep_rate=lstm_input_keep_rate,
                                                  output_keep_rate=lstm_output_keep_rate)

        self.premise_lstm_last = basic_seq_lstm_premise.last
        self.hypothesis_lstm_last = basic_seq_lstm_hypothesis.last

        self.sentence_embedding_output = tf.concat(1, [self.premise_lstm_last, self.hypothesis_lstm_last,
                                                       tf.abs(self.premise_lstm_last - self.hypothesis_lstm_last),
                                                       tf.mul(self.premise_lstm_last, self.hypothesis_lstm_last)])

        layer_1_output = nn_layer(self.sentence_embedding_output,
                                  shape=[hidden_d * 4, hidden_d * 4],
                                  name='layer-1',
                                  w_init=tf.contrib.layers.xavier_initializer(uniform=True),
                                  b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                                  act=tf.nn.tanh)

        layer_2_output = nn_layer(layer_1_output,
                                  shape=[hidden_d * 4, hidden_d * 4],
                                  name='layer-2',
                                  w_init=tf.contrib.layers.xavier_initializer(uniform=True),
                                  b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                                  act=tf.nn.tanh)

        layer_3_output = nn_layer(layer_2_output,
                                  shape=[hidden_d * 4, hidden_d * 4],
                                  name='layer-3',
                                  w_init=tf.contrib.layers.xavier_initializer(uniform=True),
                                  b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                                  act=tf.nn.tanh,
                                  output_keep=softmax_keeprate)

        self.output = nn_layer(layer_3_output,
                               shape=[hidden_d * 4, num_class],
                               name='softmax-affine-layer',
                               w_init=tf.contrib.layers.xavier_initializer(uniform=False),
                               b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                               act=None)

        self.softmax_output = tf.nn.softmax(self.output)
        self.prediction = tf.argmax(self.softmax_output, dimension=1)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.input_loader.label)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()

    def load_embedding(self, embedding=None):
        if embedding is None:
            print('No embedding specified. Use random embedding.')
        else:
            print('Load embedding.', 'Vocabulary size:', embedding.shape[0], 'Word dimension', embedding.shape[1])
            self.input_loader.load_embedding(self.sess, embedding)

    def train(self, feed_dict):
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, feed_dict):
        y_pred = feed_dict[self.input_loader.label]
        out_pred, out_cost = self.sess.run((self.prediction, self.cost), feed_dict=feed_dict)
        accuracy = np.mean(y_pred == out_pred)
        return accuracy, np.mean(out_cost)

    def setup(self, embedding=None):
        self.load_embedding(embedding=embedding)
        self.sess.run(self.init_op)
        newinfo = record_info(Word_Dimension=embedding.shape[1],
                              Vocabluary_Size=embedding.shape[0])
        """
        Update the information about the model after load embedding.
        """
        for k, v in newinfo.items():
            self.model_info[k] = v

    def close(self):
        self.sess.close()

    def test(self, feed_dict=None):
        self.setup()
        d_pred = self.sess.run(self.prediction, feed_dict=feed_dict)
        print(d_pred)
