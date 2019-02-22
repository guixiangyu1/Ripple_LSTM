import tensorflow as tf


class RippleLSTM():

    def __init__(self, config):
        self.config = config

        self.output_fw = None
        self.output_bw = None
        self.output_win = None
        self.output = None

        self.word_embeddings = None
        self.sequence_length = None


    def build(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        cell_win = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length = self.sequence_length, dtype=tf.float32)



