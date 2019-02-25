import numpy as np
import os
import tensorflow as tf

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from .fw_lstm import FwLSTM
from .bw_lstm import BwLSTM
from .win_lstm import WinLSTM


class RippleModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(RippleModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_action = {idx: action for action, idx in
                              self.config.vocab_actions.items()}
        self.FwLSTM = FwLSTM()
        self.BwLSTM = BwLSTM()
        self.WinLSTM = WinLSTM()

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""

        # placeholder会在稍后注入实际的东西
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        self.actions = tf.placeholder(tf.int32, shape=[None,None],
                                      name="actions")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)  # zip参数要求是iterable即可(batch(sentence[char]))
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # 两种数据格式
        # word_ids  [[1,2,3,4,5,0,0,0], [1,2,4,0,0,0,0,0], [1,3,4,5,6,7,8,9]]  sequence_lengths [5,3,8]
        # char_ids  [([1,2,3,0,0,0],[1,2,4,5,0,0],[0,0,0,0,0,0]),(),()] word_length ([3,4,0],[..,..,..],[])

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32)
                # trainable=self.config.train_embeddings)
            # 已经没有文字了，只有word_id
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")
            # [batchs_size, max sentence length, word_em_dim]

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                # embedding_lookup是将原来tensor中的id替换成id对应的向量
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")
                # [batch_size, max sentence length, max word length, char_em_dim]

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)
                # bidirectional_dynamic_rnn的输出为tuple，（outputs, output_states）

                # output = (output_fw, output_bw)
                # output_fw or output_bw .shape = [batchsize, max_time, cell.output_size]

                # output_states = (output_state_fw, output_state_bw)
                # output_state_fw和output_state_bw的类型为LSTMStateTuple
                # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                # output_states为(output_state_fw, output_state_bw)，包含了前向和后向 最后的 隐藏状态的组成的元组。
                # output_state_fw和output_state_bw的类型为LSTMStateTuple。
                # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
                output = tf.concat([output_fw, output_bw], axis=-1)
                # before reshape, the shape is [batchsize*sentence_length, 2* char_hidden_size]
                # after rehsape,  shape = (batch size, max sentence length, 2*char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)  # -1就是把最里面的原子元素做拼接
                # 每个word输出一个char level的embedding，然后与word level的embedding做拼接
                # 感觉作者写错了output的位置，都取得最后一个output,注意state_of_tuple=True
                # 问题得到解决：state包含最后的输出，而这里我们只需要最后一个cell的输出即可

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw],
                               axis=-1)  # shape = [batch_size, max_sentence_length,2*hidden_size_lstm]
            output = tf.nn.dropout(output, self.dropout)
        # sequence length 很重要

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:  # labels_predict=[batch_size,nstep]
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)  # [batch_size, max_sentence_length]，不需要指定最大长度
            losses = tf.boolean_mask(losses, mask)  # tf.sequence_mask和tf.boolean_mask 来对于序列的padding进行去除的内容
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    # def segment_data(self):
    def add_boundary(self):
        with tf.variable_scope("boundary"):
            self.begin = tf.get_variable(
                name="<begin>",
                dtype=tf.float32,
                shape=[self.config.dim_word]
            )
            self.end = tf.get_variable(
                name="<end>",
                dtype=tf.float32,
                shape=[self.config.dim_word]
            )
            if self.config.use_chars:
                self.char_of_begin = tf.get_variable(
                    name="begin_char",
                    dtype=tf.float32,
                    shape=[self.config.hidden_size_char]
                )
                self.char_of_begin = tf.get_variable(
                    name="end_char",
                    dtype=tf.float32,
                    shape=[self.config.hidden_size_char]
                )


    def build(self, mode=None):
        # NER specific functions
        if mode=="train":
            self.add_placeholders()

            self.add_boundary()
            for time_step in range(self.actions.shape[-1]):
                if time_step == 0:
                    self.FwLSTM.




        self.add_placeholders()

        self.segment_data()

        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip, mode=mode)

        self.initialize_session(mode=mode)  # now self.sess is defined and vars are init

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                       self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        print("p: ", p)
        print("r: ", r)

        return {"acc": 100 * acc, "f1": 100 * f1}

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
