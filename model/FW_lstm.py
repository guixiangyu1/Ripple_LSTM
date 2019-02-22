import tensorflow as tf



class FW_lstm():


    def __init__(self, config):
        self.config = config
        self.cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        self.state = self.cell_fw.zero_state(batch_size=self.config.batch_size,dtype=tf.float32)
        self.word_length = None
        self.imput = None

    # The simplest form of RNN network generated is:
    #
    # state = cell.zero_state(...)
    # outputs = []
    # for input_ in inputs:
    #     output, state = cell(input_, state)
    #     outputs.append(output)
    # return (outputs, state)
    def push(self):
        _output = tf.nn.dynamic_rnn(
            self.cell_fw, self.input, initial_state= self.state,
            sequence_length=self.word_length, dtype=tf.float32)
        _, self.state = _output
        _, hidden_state = self.state
        return hidden_state

    # 测试总结：
    # 1. dynamic_rnn的输出output=（ndarray,LSTMStateTuple）
    # 2. ndarray代表了所有的输出，为[nbatch, nstep, nhidden], dynamic_rnn的输入其实都已经补齐了，因此输出也是整齐的，
    #    只不过输出中一大堆没用的部分，这些没用的部分最终是用mask抹去
    # 3. LSTMStateTuple (c,h); c为cell状态，h为末端最后输出；二者均为最后一个lstmcell的状态
    # 4. sequence_length = 0时, 对应的输出为0，但是对应的状态为输入的状态不变

######################测试目标：sequence_length 为0时，output的输出是什么？######
    # tf.contrib.rnn.LSTMStateTuple()
    # 意外的结果：输出全部为0，状态完全由initia_state决定，与之相同