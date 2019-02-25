import tensorflow as tf



class BwLSTM():


    def __init__(self):
        self.cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        self.state = self.cell_fw.zero_state(batch_size=self.config.batch_size,dtype=tf.float32)
        self.sequence_length = None
        self.input = None
        self.output = None

    def add_all_words(self):
        self.output = tf.nn.dynamic_rnn(
            self.cell_bw, self.input, sequence_length=self.sequence_length, dtype=tf.float32
        )
        output, _ = self.output
        return output[:,:,]

    def pop(self, input):
        _output = tf.nn.dynamic_rnn(
            self.cell_fw, input, initial_state= self.state,
            sequence_length=self.word_length, dtype=tf.float32)
        _, self.state = _output
        _, hidden_state = self.state
        return hidden_state