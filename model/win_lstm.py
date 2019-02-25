import tensorflow as tf


class WinLSTM():

    def __init__(self):

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

    # tf.reverse_sequence(
    #     input,
    #     seq_lengths,
    #     seq_axis=None,
    #     batch_axis=None,
    #     name=None,
    #     seq_dim=None,
    #     batch_dim=None
    # )
    # 以上， seq_axis与seq_dim是同一个东西
    #        batch_axis与batch_dim是同一个东西
    # seq_lengths的长度应为batch_dim对应那一维的维数
    tf.reverse_sequence

