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
        self.fw_word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="fw_word_ids")
        self.bw_word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                          name="bw_word_ids")
        self.wd_word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                          name="wd_word_ids")

        # shape = (batch size)
        self.fw_sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                        name="fw_sequence_lengths")
        self.bw_sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                        name="bw_sequence_lengths")
        self.wd_sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                        name="wd_sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.fw_char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                          name="fw_char_ids")
        self.bw_char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                          name="bw_char_ids")
        self.wd_char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                          name="wd_char_ids")

        # shape = (batch_size, max_length of sentence)
        self.fw_word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                        name="fw_word_lengths")
        self.bw_word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                        name="bw_word_lengths")
        self.wd_word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                        name="wd_word_lengths")

        # shape = (batch size, max length of sentence in batch)
        # self.labels = tf.placeholder(tf.int32, shape=[None, None],
        #                              name="labels")

        # shape = (batch size)
        self.actions = tf.placeholder(tf.int32, shape=[None],
                                      name="actions")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

        # self.word_ids_reverse = tf.reverse_sequence(self.word_ids, seq_lengths=self.sequence_lengths, batch_axis=0, seq_axis=-1)
        # self.char_ids_reverse = tf.reverse_sequence(self.char_ids, seq_lengths=self.sequence_lengths, batch_axis=0, seq_axis=1)

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
                self._word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                self._word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32)
                # trainable=self.config.train_embeddings)
            # 已经没有文字了，只有word_id
            fw_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings,
                                                        self.fw_word_ids, name="fw_word_embeddings")
            bw_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings,
                                                        self.bw_word_ids, name="bw_word_embeddings")
            wd_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings,
                                                        self.wd_word_ids, name="wd_word_embeddings")
            # [batchs_size, max sentence length, word_em_dim]


        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                self._char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                # embedding_lookup是将原来tensor中的id替换成id对应的向量
                fw_char_embeddings = tf.nn.embedding_lookup(self._char_embeddings,
                                                            self.fw_char_ids, name="fw_char_embeddings")
                bw_char_embeddings = tf.nn.embedding_lookup(self._char_embeddings,
                                                            self.bw_char_ids, name="bw_char_embeddings")
                wd_char_embeddings = tf.nn.embedding_lookup(self._char_embeddings,
                                                            self.wd_char_ids, name="wd_char_embeddings")
                # [batch_size, max sentence length, max word length, char_em_dim]

                # put the time dimension on axis=1
                s_fw = tf.shape(fw_char_embeddings)
                fw_char_embeddings = tf.reshape(fw_char_embeddings,
                                             shape=[s_fw[0] * s_fw[1], s_fw[-2], self.config.dim_char])
                fw_word_lengths = tf.reshape(self.fw_word_lengths, shape=[s_fw[0] * s_fw[1]])

                s_bw = tf.shape(bw_char_embeddings)
                bw_char_embeddings = tf.reshape(bw_char_embeddings,
                                                shape=[s_bw[0] * s_bw[1], s_bw[-2], self.config.dim_char])
                bw_word_lengths = tf.reshape(self.bw_word_lengths, shape=[s_bw[0] * s_bw[1]])

                s_wd = tf.shape(wd_char_embeddings)
                wd_char_embeddings = tf.reshape(wd_char_embeddings,
                                                shape=[s_fw[0] * s_fw[1], s_fw[-2], self.config.dim_char])
                wd_word_lengths = tf.reshape(self.wd_word_lengths, shape=[s_wd[0] * s_wd[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                _fw_output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, fw_char_embeddings,
                    sequence_length=fw_word_lengths, dtype=tf.float32)
                _, ((_, fw_output_fw), (_, fw_output_bw)) = _fw_output

                _bw_output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, bw_char_embeddings,
                    sequence_length=bw_word_lengths, dtype=tf.float32)
                _, ((_, bw_output_fw), (_, bw_output_bw)) = _bw_output

                _wd_output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, wd_char_embeddings,
                    sequence_length=wd_word_lengths, dtype=tf.float32)
                _, ((_, wd_output_fw), (_, wd_output_bw)) = _wd_output
                # output_states为(output_state_fw, output_state_bw)，包含了前向和后向 最后的 隐藏状态的组成的元组。
                # output_state_fw和output_state_bw的类型为LSTMStateTuple。
                # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
                fw_output = tf.concat([fw_output_fw, fw_output_bw], axis=-1)
                bw_output = tf.concat([bw_output_fw, bw_output_bw], axis=-1)
                wd_output = tf.concat([wd_output_fw, wd_output_bw], axis=-1)
                # before reshape, the shape is [batchsize*sentence_length, 2* char_hidden_size]
                # after rehsape,  shape = (batch size, max sentence length, 2*char hidden size)
                fw_output = tf.reshape(fw_output,
                                       shape=[s_fw[0], s_fw[1], 2 * self.config.hidden_size_char])
                bw_output = tf.reshape(bw_output,
                                       shape=[s_bw[0], s_bw[1], 2 * self.config.hidden_size_char])
                wd_output = tf.reshape(wd_output,
                                       shape=[s_wd[0], s_wd[1], 2 * self.config.hidden_size_char])
                fw_word_embeddings = tf.concat([fw_word_embeddings, fw_output], axis=-1)
                bw_word_embeddings = tf.concat([bw_word_embeddings, bw_output], axis=-1)
                wd_word_embeddings = tf.concat([wd_word_embeddings, wd_output], axis=-1)

        begin_word_embeddings = tf.get_variable("begin_word_embedding",
            shape=[self.config.batch_size, 1, 2*self.config.hidden_size_char+self.config.dim_word], dtype=tf.float32)
        end_word_embeddings = tf.get_variable("end_word_embedding",
            shape=[self.config.batch_size, 1, 2*self.config.hidden_size_char+self.config.dim_word], dtype=tf.float32)

        # reverse backward embeddings
        bw_word_embeddings = tf.reverse_sequence(bw_word_embeddings,self.bw_sequence_lengths,
                                                 seq_axis=1, batch_axis=0)


        fw_word_embeddings = tf.concat([begin_word_embeddings, fw_word_embeddings], axis=1)
        bw_word_embeddings = tf.concat([end_word_embeddings, bw_word_embeddings], axis=1)
        self.fw_word_embeddings = tf.nn.dropout(fw_word_embeddings, self.dropout)
        self.bw_word_embeddings = tf.nn.dropout(bw_word_embeddings, self.dropout)
        self.wd_word_embeddings = tf.nn.dropout(wd_word_embeddings, self.dropout)

        self.fw_sequence_lengths = self.fw_sequence_lengths + tf.ones([self.config.batch_size], dtype=tf.int32)
        self.bw_sequence_lengths = self.bw_sequence_lengths + tf.ones([self.config.batch_size], dtype=tf.int32)



        # self.begin_word_embeddings = tf.nn.dropout(begin_word_embeddings, self.dropout)
        # self.end_word_embeddings   = tf.nn.dropout(end_word_embeddings,   self.dropout)





    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """

        with tf.variable_scope("lstm"):
            fw_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            bw_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            wd_cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_win)
            wd_cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_win)



            _, state_fw_begin = tf.nn.dynamic_rnn(
                cell_fw, self.begin_embedding, dtype=tf.float32
            )
            _, state_bw_end = tf.nn.dynamic_rnn(
                cell_bw, self.end_embedding, dtype=tf.float32
            )


            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                initial_state_fw=state_fw_begin, initial_state_bw=state_bw_end,
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




    def get_output(self,output,point):
        '''

        :param output: size [batch_size, max_length, hidden_dim]
        :param point: [batch_size]
        :return:
        '''
        batch_num = tf.reshape(tf.range(0,self.config.batch_size,dtype=tf.int32),[self.config.batch_size,1])
        indices = tf.concat([batch_num, tf.reshape(point,[self.config.batch_size,1])], -1)
        return tf.gather_nd(output,indices)


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
    def add_boundary_embedding(self):
        with tf.variable_scope("boundary"):
            embedding_begin = tf.get_variable(
                name="<embedding_for_begin>",
                dtype=tf.float32,
                shape=[self.config.batch_size, 1, self.config.dim_word]
            )
            embedding_end = tf.get_variable(
                name="<embedding_for_end>",
                dtype=tf.float32,
                shape=[self.config.batch_size, 1, self.config.dim_word]
            )
            if self.config.use_chars:
                embedding_charbegin = tf.get_variable(
                    name="enbedding_for_beginchar",
                    dtype=tf.float32,
                    shape=[self.config.batch_size, 1, self.config.hidden_size_char*2]
                )
                embedding_charend = tf.get_variable(
                    name="embedding_for_endchar",
                    dtype=tf.float32,
                    shape=[self.config.batch_size, 1, self.config.hidden_size_char*2]
                )
                embedding_begin = tf.concat([embedding_begin, embedding_charbegin], -1)
                embedding_end   = tf.concat([embedding_end, embedding_charend], -1)
        self.begin_embedding = tf.nn.dropout(embedding_begin, self.dropout)
        self.end_embedding   = tf.nn.dropout(embedding_end,   self.dropout)


    def build(self, mode=None):
        # NER specific functions

        self.add_placeholders()
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


