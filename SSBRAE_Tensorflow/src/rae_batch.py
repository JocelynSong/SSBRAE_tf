import tensorflow as tf
import numpy as np
import logging

from src.utils import get_variable
from src.similarity import cosine_similarity_batch
from src.data_utils.phrase_utils import WORD_INDEX, PARA_INDEX, TRAN_INDEX
from src import OOV_KEY
from src.Initializer import data_type


logger = logging.getLogger(__name__)


def sem_distance(p1, w1, b1, p2):
    """
    compute the distance of transformed source and target phrases
    :param p1:
    :param w1:
    :param b1:
    :param p2:
    :return:
    """
    transformed_p1 = tf.tanh(tf.matmul(p1, w1) + b1)
    return tf.reduce_sum(tf.square(transformed_p1 - p2), axis=1) / 2


def bilinear_score(p1, w, p2):
    """
    compute the bilinear score of two phrase embeddings
    :param p1:
    :param w:
    :param p2:
    :return:
    """
    transformed_p1 = tf.matmul(p1, w)
    return tf.reduce_sum(transformed_p1 * p2, axis=1)


def sem_sim_distance(p1, w1, b1, p2):
    transformed_p1 = tf.tanh(tf.matmul(p1, w1) + b1)
    return cosine_similarity_batch(transformed_p1, p2)


def conditional_probabilities(p1, w1, b1, p2):
    """
    compute the conditional probability
    :param p1:
    :param w1:
    :param b1:
    :param p2:
    :return:
    """
    transformed_p1 = tf.tanh(tf.matmul(p1, w1) + b1)
    scores = tf.exp(-tf.reduce_sum(tf.square(transformed_p1 - p2), axis=1))
    return scores


class RAEEncoder(object):
    def __init__(self, activation, embedding, normalize, weight_rec, weight_embedding, n_epoch, max_src_len,
                 batch_size, dropout, optimizer_config, sess, name=""):
        self.activation = activation
        self.embedding = embedding
        self.dim = self.embedding.dim
        self.normalize = normalize
        self.weight_rec = weight_rec
        self.weight_embedding = weight_embedding
        self.max_src_len = max_src_len
        # self.max_tar_len = max_tar_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.dropout = dropout
        self.sess = sess
        self.optimizer = optimizer_config.optimizer
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("rae_encoder"):
            self.W = get_variable(shape=(self.dim * 2, self.dim), name="rae_w")
            self.b = get_variable(shape=(self.dim,), name="rae_b")
            self.Wr = get_variable(shape=(self.dim, self.dim * 2), name="rae_wr")
            self.br = get_variable(shape=(self.dim * 2,), name="rae_br")
        self.inputs = []
        for i in range(self.batch_size):
            self.inputs.append(tf.placeholder(tf.int32, shape=[self.max_src_len],
                                              name=name+"pos_encoder{0}".format(i)))
        # self.inputs = tf.transpose(self.inputs)

        self.params = [self.W, self.b, self.Wr, self.br]
        self.l1_norm = tf.reduce_sum([tf.reduce_sum(tf.abs(param)) for param in self.params])
        self.l2_norm = tf.reduce_sum([tf.reduce_sum(tf.square(param)) for param in self.params])
        self.loss_l2_embedding = self.weight_embedding * self.embedding.l2_norm
        self.loss_l2_rec = self.weight_rec * self.l2_norm
        self.rec_loss = self.compose() / self.batch_size
        self.loss = self.rec_loss + self.loss_l2_embedding + self.loss_l2_rec

        self.sess.run(tf.global_variables_initializer())
        self.train = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def embedding_index_assign(self, phrase_embedding, x, y):
        opt = phrase_embedding[:, y, :].assign(phrase_embedding[:, x, :])
        self.sess.run(opt)
        return phrase_embedding

    def embedding_assign(self, phrase_embedding, x, embedding):
        opt = phrase_embedding[:, x, :].assign(embedding)
        self.sess.run(opt)
        return phrase_embedding

    '''
    def encode_step(self, phrase_embedding, i, end):
        total_embedding = tf.concat([phrase_embedding[:, i, :], phrase_embedding[:, end, :]], axis=2)
        total_embedding = tf.reshape(total_embedding, [self.batch_size, self.dim * 2])
        result = self.activation.activate(tf.matmul(total_embedding, self.W) + self.b)
        result = tf.reshape(result, [self.batch_size, 1, self.dim])
        phrase_embedding = self.embedding_assign(phrase_embedding, end, result)
        return phrase_embedding

    def encode(self, phrase_embedding):
        self.embedding_index_assign(phrase_embedding, 0, self.max_src_len)
        for i in range(1, self.max_src_len):
            phrase_embedding = self.encode_step(phrase_embedding, i, self.max_src_len)
        return phrase_embedding
    '''

    def encode_step(self, phrase_embedding, i, result_emb):
        #total_embedding = tf.concat([phrase_embedding[:, i, :], result_emb], axis=2)
        left = tf.reshape(phrase_embedding[:, i, :], [self.batch_size, self.dim])
        right = tf.reshape(result_emb, [self.batch_size, self.dim])
        total_embedding = tf.concat([left, right], axis=1)
        result = self.activation.activate(tf.matmul(total_embedding, self.W) + self.b)
        result = tf.reshape(result, [self.batch_size, 1, self.dim])
        return result

    def encode(self, phrase_embedding):
        result_emb = phrase_embedding[:, 0, :]
        for i in range(1, self.max_src_len):
            result_emb = self.encode_step(phrase_embedding, i, result_emb)
        return result_emb

    def decode_step(self, compose_embedding):
        now_embedding = tf.reshape(compose_embedding, [self.batch_size, self.dim])
        result = self.activation.activate(tf.matmul(now_embedding, self.Wr) + self.br)
        result = tf.reshape(result, [self.batch_size, 1, self.dim * 2])
        left_embedding = tf.reshape(result[:, 0, :self.dim], [self.batch_size, 1, self.dim])
        right_embedding = tf.reshape(result[:, 0, self.dim:], [self.batch_size, 1, self.dim])
        return left_embedding, right_embedding

    def decode(self, com_embedding):
        result = []
        for i in range(1, self.max_src_len):
            com_embedding, right_result = self.decode_step(com_embedding)
            result.append(right_result)
        result.append(com_embedding)
        result_emb = tf.concat([result[len(result)-1-i] for i in range(len(result))], axis=1)
        return result_emb

    def compose(self):
        phrase_embeddings = tf.nn.embedding_lookup(self.embedding.W, self.inputs) #[batch_size, max_len+1, dim]
        print(phrase_embeddings)
        com_result = self.encode(phrase_embeddings)
        rec_embeddings = self.decode(com_result)
        phrase_embeddings = tf.nn.embedding_lookup(self.embedding.W, self.inputs)
        loss = tf.reduce_sum(tf.square(rec_embeddings - phrase_embeddings), axis=2)
        #m_ini = np.ones([self.max_src_len + 1])
        #m_ini[self.max_src_len] = 0
        #mask = tf.Variable(m_ini, dtype=tf.float32, trainable=False)
        #loss = loss * mask
        rec_loss = tf.reduce_sum(loss) / 2.0
        return rec_loss

    def output(self, inputs):
        phrase_embeddings = tf.nn.embedding_lookup(self.embedding.W, inputs)
        com_result = self.encode(phrase_embeddings)
        #compose_embedding = com_result[:, self.max_src_len, :]
        compose_embedding = tf.reshape(com_result, [self.batch_size, self.dim])
        return compose_embedding

    def train_step(self, inputs):
        input_feed = {}
        for i in range(self.batch_size):
            input_feed[self.inputs[i].name] = inputs[i]

        output_feed = [self.train, self.loss]

        _, loss = self.sess.run(output_feed, input_feed)
        return loss

    def predict_step(self, inputs):
        input_feed = {}
        for i in range(self.batch_size):
            input_feed[self.inputs[i].name] = inputs[i]

        output_feed = self.loss

        loss = self.sess.run(output_feed, input_feed)
        return loss

    def get_batch(self, phrases_list, train_index, i, word_idx):
        oov_idx = word_idx[OOV_KEY]
        inputs = []
        for j in range(self.batch_size):
            phrase = phrases_list[train_index[i * self.batch_size + j]]
            words_list = phrase[WORD_INDEX]
            if len(words_list) < self.max_src_len:
                for i in range(self.max_src_len - len(words_list)):
                    words_list.append(oov_idx)
            inputs.append(words_list)
        return inputs


class NegativeRAEEncoder(RAEEncoder):
    def __init__(self, activation, embedding, normalize, weight_rec, weight_embedding, n_epoch, max_src_len,
                 batch_size, dropout, optimizer_config, sess, name):
        super(NegativeRAEEncoder, self).__init__(activation, embedding, normalize, weight_rec, weight_embedding,
                                                 n_epoch, max_src_len, batch_size, dropout, optimizer_config, sess, name)
        self.neg_inputs = []
        for i in range(self.batch_size):
            self.neg_inputs.append(tf.placeholder(dtype=tf.int32, shape=[self.max_src_len],
                                                  name=name+"neg_encoder{0}".format(i)))

    def neg_output(self, neg_inputs):
        phrase_embeddings = tf.nn.embedding_lookup(self.embedding.W, neg_inputs)
        com_result = self.encode(phrase_embeddings)
        #compose_embedding = com_result[:, self.max_src_len, :]
        compose_embedding = tf.reshape(com_result, [self.batch_size, self.dim])
        return compose_embedding


class BilingualPhraseRAE(object):
    def __init__(self, src_embeddings, tar_embeddings, activation, normalize, weight_rec, weight_sem, weight_embedding,
                 alpha, max_src_len, max_tar_len, n_epoch, batch_size, dropout, optimizer_config, sess):
        self.src_embeddings = src_embeddings
        self.tar_embeddings = tar_embeddings
        self.activation = activation
        self.dim = src_embeddings.dim
        self.weight_sem = weight_sem
        self.alpha = alpha
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer_config.optimizer
        self.sess = sess
        #self.global_step = tf.Variable(0, trainable=False)

        self.src_encoder = NegativeRAEEncoder(activation, src_embeddings, normalize, weight_rec, weight_embedding,
                                              n_epoch, max_src_len, batch_size, dropout, optimizer_config, sess, "src_")
        self.tar_encoder = NegativeRAEEncoder(activation, tar_embeddings, normalize, weight_rec, weight_embedding,
                                              n_epoch, max_tar_len, batch_size, dropout, optimizer_config, sess, "tar_")
        self.src_pos_output = self.src_encoder.output(self.src_encoder.inputs)
        self.src_neg_output = self.src_encoder.neg_output(self.src_encoder.neg_inputs)
        self.tar_pos_output = self.tar_encoder.output(self.tar_encoder.inputs)
        self.tar_neg_output = self.tar_encoder.neg_output(self.tar_encoder.neg_inputs)

        with tf.variable_scope("bilingual_rae_encoder"):
            self.W = get_variable([self.dim, self.dim], name="bilingual_w")
            self.b = get_variable([self.dim], name="bilingual_b")
            self.Wr = get_variable([self.dim, self.dim], name="bilingual_wr")
            self.br = get_variable([self.dim], name="bilingual_br")

        # positive instances distance
        self.src_pos_sem = sem_distance(self.src_pos_output, self.W, self.b, self.tar_pos_output)
        self.tar_pos_sem = sem_distance(self.tar_pos_output, self.Wr, self.br, self.src_pos_output)
        # negative instances distance
        self.src_neg_sem = sem_distance(self.src_pos_output, self.W, self.b, self.tar_neg_output)
        self.tar_neg_sem = sem_distance(self.tar_pos_output, self.Wr, self.br, self.src_neg_output)
        # max margin error
        self.src_max_margin = tf.maximum(0.0, self.src_pos_sem - self.src_neg_sem + 1.0)
        self.tar_max_margin = tf.maximum(0.0, self.tar_pos_sem - self.tar_neg_sem + 1.0)

        # similarity error
        self.loss_sem = (tf.reduce_sum(self.src_max_margin) + tf.reduce_sum(self.tar_max_margin)) / self.batch_size
        self.loss_rec = self.src_encoder.rec_loss + self.tar_encoder.rec_loss

        params = [self.W, self.b, self.Wr, self.br]
        self.l1_norm = tf.reduce_sum([tf.reduce_sum(tf.abs(param)) for param in params])
        self.l2_norm = tf.reduce_sum([tf.reduce_sum(tf.square(param)) for param in params])
        self.loss_l2_sem = self.weight_sem * self.l2_norm
        self.loss_l2_rec = self.src_encoder.loss_l2_rec + self.tar_encoder.loss_l2_rec
        self.loss_l2_embedding = self.src_encoder.loss_l2_embedding + self.tar_encoder.loss_l2_embedding
        self.loss_l2 = self.loss_l2_embedding + self.loss_l2_rec + self.loss_l2_sem

        self.sess.run(tf.global_variables_initializer())
        self.loss = self.alpha * self.loss_rec + (1-self.alpha) * self.loss_sem + self.loss_l2
        self.train = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def train_step(self, src_pos, tar_pos, src_neg, tar_neg):
        input_feed = {}

        for i in range(self.batch_size):
            input_feed[self.src_encoder.inputs[i].name] = src_pos[i]
            input_feed[self.tar_encoder.inputs[i].name] = tar_pos[i]
            input_feed[self.src_encoder.neg_inputs[i].name] = src_neg[i]
            input_feed[self.tar_encoder.neg_inputs[i].name] = tar_neg[i]

        output_feed = [self.train, self.loss]
        _, loss = self.sess.run(output_feed, input_feed)
        return loss

    def predict_step(self, src_pos, tar_pos, src_neg, tar_neg):
        input_feed = {}

        for i in range(self.batch_size):
            input_feed[self.src_encoder.inputs[i].name] = src_pos[i]
            input_feed[self.tar_encoder.inputs[i].name] = tar_pos[i]
            input_feed[self.src_encoder.neg_inputs[i].name] = src_neg[i]
            input_feed[self.tar_encoder.neg_inputs[i].name] = tar_neg[i]

        output_feed = self.loss
        loss = self.sess.run(output_feed, input_feed)
        return loss

    def fill_phrase_with_oov(self, phrase, oov_key, src=True):# src=1 for src phrase, else for tar phrase
        if src:
            for i in range(self.max_src_len - len(phrase)):
                phrase.append(oov_key)
        else:
            for i in range(self.max_tar_len - len(phrase)):
                phrase.append(oov_key)
        return phrase

    def get_batch(self, src_phrases, tar_phrases, bi_phrase_list, bi_phrase_train_index, src_word_idx, tar_word_idx, k):
        src_oov = src_word_idx[OOV_KEY]
        tar_oov = tar_word_idx[OOV_KEY]

        src_pos = []
        tar_pos = []
        src_neg = []
        tar_neg = []
        for i in range(self.batch_size):
            src_idx, tar_idx = bi_phrase_list[bi_phrase_train_index[k * self.batch_size + i]]
            src_words_ind = src_phrases[src_idx][WORD_INDEX]
            tar_words_ind = tar_phrases[tar_idx][WORD_INDEX]

            src_neg_i = np.random.randint(0, len(src_phrases))
            tar_neg_i = np.random.randint(0, len(tar_phrases))
            src_neg_words_ind = src_phrases[src_neg_i][WORD_INDEX]
            tar_neg_words_ind = tar_phrases[tar_neg_i][WORD_INDEX]

            src_pos.append(self.fill_phrase_with_oov(src_words_ind, src_oov, True))
            src_neg.append(self.fill_phrase_with_oov(src_neg_words_ind, src_oov, True))
            tar_pos.append(self.fill_phrase_with_oov(tar_words_ind, tar_oov, False))
            tar_neg.append(self.fill_phrase_with_oov(tar_neg_words_ind, tar_oov, False))

        return src_pos, tar_pos, src_neg, tar_neg


class SSBRAEEncoder(BilingualPhraseRAE):
    def __init__(self, src_embeddings, tar_embeddings, activation, normalize, weight_rec, weight_sem,
                 weight_embedding, alpha, beta, max_src_len, max_tar_len, n_epoch, batch_size, dropout,
                 optimizer_config, para, trans, para_num, trans_num, sess):
        super(SSBRAEEncoder, self).__init__(src_embeddings, tar_embeddings, activation, normalize, weight_rec,
                                            weight_sem, weight_embedding, alpha, max_src_len, max_tar_len,
                                            n_epoch, batch_size, dropout, optimizer_config, sess)
        self.beta = beta
        self.para = para
        self.trans = trans
        self.para_num = para_num
        self.trans_num = trans_num

        # para
        self.src_para = []
        self.src_para_weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.para_num],
                                              name="src_para_weight")
        self.tar_para = []
        self.tar_para_weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.para_num],
                                              name="tar_para_weight")

        # trans
        self.src_tar_trans = []
        self.src_tar_trans_weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.trans_num],
                                                   name="trans_src_tar_weight")
        self.tar_src_trans = []
        self.tar_src_trans_weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.trans_num],
                                                   name="trans_tar_src_weight")

        for i in range(self.batch_size):
            self.src_para.append([])
            self.tar_para.append([])
            #self.src_para[i] = []
            #self.tar_para[i] = []
            for j in range(self.para_num):
                self.src_para[i].append(tf.placeholder(tf.int32, shape=[self.max_src_len],
                                                       name="para_src%d%d" % (i, j)))
                self.tar_para[i].append(tf.placeholder(tf.int32, shape=[self.max_tar_len]))

            #self.src_tar_trans[i] = []
            #self.tar_src_trans[i] = []
            self.src_tar_trans.append([])
            self.tar_src_trans.append([])
            for j in range(self.trans_num):
                self.src_tar_trans[i].append(tf.placeholder(tf.int32, shape=[self.max_tar_len],
                                                            name="trans_src_tar%d%d" % (i, j)))
                self.tar_src_trans[i].append(tf.placeholder(tf.int32, shape=[self.max_src_len],
                                                            name="trans_tar_src%d%d" % (i, j)))

        if self.para:
            self.src_para_output = []
            self.tar_para_output = []
            for i in range(self.para_num):
                # (batch_size, dim)
                src_inputs = []
                tar_inputs = []
                for k in range(self.batch_size):
                    src_inputs.append(self.src_para[k][i])
                    tar_inputs.append(self.tar_para[k][i])
                self.src_para_output.append(self.src_encoder.output(src_inputs))
                self.tar_para_output.append(self.tar_encoder.output(tar_inputs))
            self.loss_para_src = self.get_para_loss(self.src_pos_output, self.src_para_output, self.src_para_weight)
            self.loss_para_tar = self.get_para_loss(self.tar_pos_output, self.tar_para_output, self.tar_para_weight)
            self.loss_para = (self.loss_para_src + self.loss_para_tar) / self.batch_size

        if self.trans:
            self.src_tar_trans_output = []
            self.tar_src_trans_output = []
            for i in range(self.trans_num):
                src_tar_inputs = []
                tar_src_inputs = []
                for k in range(self.batch_size):
                    src_tar_inputs.append(self.src_tar_trans[k][i])
                    tar_src_inputs.append(self.tar_src_trans[k][i])
                self.src_tar_trans_output.append(self.tar_encoder.output(src_tar_inputs))
                self.tar_src_trans_output.append(self.src_encoder.output(tar_src_inputs))
            self.loss_trans_src_tar = self.get_trans_loss(self.src_pos_output, self.src_tar_trans_output,
                                                          self.W, self.b, self.src_tar_trans_weight)
            self.loss_trans_tar_src = self.get_trans_loss(self.tar_pos_output, self.tar_src_trans_output,
                                                          self.Wr, self.br, self.tar_src_trans_weight)
            self.loss_trans = (self.loss_trans_src_tar + self.loss_trans_tar_src) / self.batch_size

        self.loss = self.loss_l2 + self.alpha * self.loss_rec + self.beta * self.loss_sem
        if self.para and not self.trans:
            self.loss += (1 - self.alpha - self.beta) * self.loss_para
        elif self.trans and not self.para:
            self.loss += (1 - self.alpha - self.beta) * self.loss_trans
        else:
            self.loss += (1 - self.alpha - self.beta) * (self.loss_para + self.loss_trans)
        self.train = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def get_para_loss(self, pos_inputs, para_inputs, weights):
        scores = tf.exp(-tf.reduce_sum(tf.square(pos_inputs - para_inputs), axis=1))
        scores = tf.reshape(scores, [self.para_num, self.batch_size])
        total_score = tf.reduce_sum(scores, axis=0)
        values = tf.transpose(scores / total_score)  # [batch_size, para_num]
        loss = tf.reduce_sum(weights * tf.log(weights / values))
        return loss

    def get_trans_loss(self, src_inputs, tar_inputs, w, b, weights):
        scores = conditional_probabilities(src_inputs, w, b, tar_inputs)
        scores = tf.reshape(scores, [self.trans_num, self.batch_size])
        total_score = tf.reduce_sum(scores, axis=0)
        values = tf.transpose(scores / total_score)
        loss = tf.reduce_sum(weights * tf.log(weights / values))
        return loss

    def get_para_instance(self, phrases, index, oov_key, src=True):
        para_list = phrases[index][PARA_INDEX]
        para = []
        weights = []
        if len(para_list) >= self.para_num:
            for i in range(self.para_num):
                phrase_ind, weight = para_list[i]
                phrase = phrases[phrase_ind][WORD_INDEX]
                phrase = self.fill_phrase_with_oov(phrase, oov_key, src)
                para.append(phrase)
                weights.append(weight)
        else:
            for i in range(len(para_list)):
                phrase_ind, weight = para_list[i]
                phrase = phrases[phrase_ind][WORD_INDEX]
                phrase = self.fill_phrase_with_oov(phrase, oov_key, src)
                para.append(phrase)
                weights.append(weight)
            for i in range(self.para_num - len(para_list)):
                phrase = self.fill_phrase_with_oov([], oov_key, src)
                para.append(phrase)
                weights.append(0.0001)
        if np.sum(weights) != 0.:
            total = np.sum(weights)
            weights /= total
        else:
            for j in range(self.para_num):
                weights[j] = 1. / float(self.para_num)
        return para, weights

    def get_trans_instance(self, src_phrases, tar_phrases, index, oov_key, src=True):
        trans_dict = src_phrases[index][TRAN_INDEX]
        trans = []
        trans_weight = []
        if len(trans_dict) >= self.trans_num:
            i = 0
            for ind, weight in trans_dict.items():
                phrase = tar_phrases[ind][WORD_INDEX]
                phrase = self.fill_phrase_with_oov(phrase, oov_key, src)
                trans.append(phrase)
                trans_weight.append(weight)
                i += 1
                if i >= self.trans_num:
                    break
        else:
            for ind, weight in trans_dict.items():
                phrase = tar_phrases[ind][WORD_INDEX]
                phrase = self.fill_phrase_with_oov(phrase, oov_key, src)
                trans.append(phrase)
                trans_weight.append(weight)
            for i in range(self.trans_num - len(trans_dict)):
                trans.append(self.fill_phrase_with_oov([], oov_key, src))
                trans_weight.append(0.0001)
        if np.sum(trans_weight) != 0.:
            trans_weight /= np.sum(trans_weight)
        else:
            for i in range(self.trans_num):
                trans_weight[i] = 1. / float(self.trans_num)
        return trans, trans_weight

    def ssbrae_train_step(self, src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para, src_para_weight,
                          tar_para_weight, src_tar_trans, tar_src_trans, src_tar_trans_weight, tar_src_trans_weight):
        input_feed = {}
        for i in range(self.batch_size):
            input_feed[self.src_encoder.inputs[i].name] = src_pos[i]
            input_feed[self.tar_encoder.inputs[i].name] = tar_pos[i]
            input_feed[self.src_encoder.neg_inputs[i].name] = src_neg[i]
            input_feed[self.tar_encoder.neg_inputs[i].name] = tar_neg[i]
        if self.para:
            for i in range(self.batch_size):
                for j in range(self.para_num):
                    input_feed[self.src_para[i][j].name] = src_para[i][j]
                    input_feed[self.tar_para[i][j].name] = tar_para[i][j]
            input_feed[self.src_para_weight.name] = src_para_weight
            input_feed[self.tar_para_weight.name] = tar_para_weight
        if self.trans:
            for i in range(self.batch_size):
                for j in range(self.trans_num):
                    input_feed[self.src_tar_trans[i][j].name] = src_tar_trans[i][j]
                    input_feed[self.tar_src_trans[i][j].name] = tar_src_trans[i][j]
            input_feed[self.src_tar_trans_weight.name] = src_tar_trans_weight
            input_feed[self.tar_src_trans_weight.name] = tar_src_trans_weight

        output_feed = [self.train, self.loss, self.loss_l2, self.loss_rec, self.loss_sem]
        if self.para and not self.trans:
            output_feed.append(self.loss_para)
        elif self.trans and not self.para:
            output_feed.append(self.loss_trans)
        elif self.para and self.trans:
            output_feed.append(self.loss_para)
            output_feed.append(self.loss_trans)
        else:
            raise ValueError("No such para and trans configuration: para%s trans%s" % (str(self.para), str(self.trans)))
        result = self.sess.run(output_feed, input_feed)
        return result

    def ssbrae_predict_step(self, src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para, src_para_weight,
                            tar_para_weight, src_tar_trans, tar_src_trans, src_tar_trans_weight, tar_src_trans_weight):
        input_feed = {}
        for i in range(self.batch_size):
            input_feed[self.src_encoder.inputs[i].name] = src_pos[i]
            input_feed[self.tar_encoder.inputs[i].name] = tar_pos[i]
            input_feed[self.src_encoder.neg_inputs[i].name] = src_neg[i]
            input_feed[self.tar_encoder.neg_inputs[i].name] = tar_neg[i]
        if self.para:
            for i in range(self.batch_size):
                for j in range(self.para_num):
                    input_feed[self.src_para[i][j].name] = src_para[i][j]
                    input_feed[self.tar_para[i][j].name] = tar_para[i][j]
            input_feed[self.src_para_weight.name] = src_para_weight
            input_feed[self.tar_para_weight.name] = tar_para_weight
        if self.trans:
            for i in range(self.batch_size):
                for j in range(self.trans_num):
                    input_feed[self.src_tar_trans[i][j].name] = src_tar_trans[i][j]
                    input_feed[self.tar_src_trans[i][j].name] = tar_src_trans[i][j]
            input_feed[self.src_tar_trans_weight.name] = src_tar_trans_weight
            input_feed[self.tar_src_trans_weight.name] = tar_src_trans_weight

        output_feed = [self.loss, self.loss_l2, self.loss_rec, self.loss_sem]
        if self.para and not self.trans:
            output_feed.append(self.loss_para)
        elif self.trans and not self.para:
            output_feed.append(self.loss_trans)
        elif self.para and self.trans:
            output_feed.append(self.loss_para)
            output_feed.append(self.loss_trans)
        else:
            raise ValueError("No such para and trans configuration: para%s trans%s" % (str(self.para), str(self.trans)))
        result = self.sess.run(output_feed, input_feed)
        return result

    def get_batch(self, src_phrases, tar_phrases, bi_phrase_list, bi_phrase_train_index, src_word_idx, tar_word_idx, k):
        src_pos, tar_pos, src_neg, tar_neg = super(SSBRAEEncoder, self).get_batch(src_phrases, tar_phrases,
                                                                                  bi_phrase_list, bi_phrase_train_index,
                                                                                  src_word_idx, tar_word_idx, k)
        src_oov = src_word_idx[OOV_KEY]
        tar_oov = tar_word_idx[OOV_KEY]

        src_para = []
        src_para_weight = []
        tar_para = []
        tar_para_weight = []

        src_tar_trans = []
        src_tar_trans_weight = []
        tar_src_trans = []
        tar_src_trans_weight = []
        for i in range(self.batch_size):
            src_ind, tar_ind = bi_phrase_list[bi_phrase_train_index[k * self.batch_size + i]]

            # src para
            src_para_instance, src_para_instance_weight = self.get_para_instance(src_phrases, src_ind, src_oov, True)
            src_para.append(src_para_instance)
            src_para_weight.append(src_para_instance_weight)

            # tar para
            tar_para_instance, tar_para_instance_weight = self.get_para_instance(tar_phrases, tar_ind, tar_oov, False)
            tar_para.append(tar_para_instance)
            tar_para_weight.append(tar_para_instance_weight)

            # src2tar trans
            src_tar_trans_instance, src_tar_trans_ins_weight = self.get_trans_instance(src_phrases, tar_phrases,
                                                                                       src_ind, tar_oov, False)
            src_tar_trans.append(src_tar_trans_instance)
            src_tar_trans_weight.append(src_tar_trans_ins_weight)

            # tar2src trans
            tar_src_trans_instance, tar_src_trans_ins_weight = self.get_trans_instance(tar_phrases, src_phrases,
                                                                                       tar_ind, src_oov, True)
            tar_src_trans.append(tar_src_trans_instance)
            tar_src_trans_weight.append(tar_src_trans_ins_weight)

        return (src_pos, tar_pos,
                src_neg, tar_neg,
                src_para, tar_para,
                src_para_weight, tar_para_weight,
                src_tar_trans, tar_src_trans,
                src_tar_trans_weight, tar_src_trans_weight)































































