import tensorflow as tf



class BW_lstm():


    def __init__(self, config):
        self.config = config
        self.cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        self.state = self.cell_fw.zero_state(batch_size=self.config.batch_size,dtype=tf.float32)
        self.word_length = None

    # The simplest form of RNN network generated is:
    #
    # state = cell.zero_state(...)
    # outputs = []
    # for input_ in inputs:
    #     output, state = cell(input_, state)
    #     outputs.append(output)
    # return (outputs, state)
    def push(self, input):
        _output = tf.nn.dynamic_rnn(
            self.cell_fw, input, initial_state= self.state,
            sequence_length=self.word_length, dtype=tf.float32)
        _, self.state = _output
        _, hidden_state = self.state
        return hidden_state