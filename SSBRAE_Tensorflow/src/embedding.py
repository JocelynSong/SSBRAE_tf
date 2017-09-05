import logging

import tensorflow as tf
import numpy as np
import sys

from src import default_initializer, OOV_KEY
from src.utils import get_variable
from src.Initializer import data_type


__author__ = "Jocelyn"
RAEL = np.float32
logger = logging.getLogger(__name__)
if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, encoding="utf-8", errors="strict"):
    """
    converts a string(bytesstring of encoding or unicode) to unicode
    :param text:
    :param encoding:
    :param errors:
    :return:
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding=encoding, errors=errors)


class Embedding(object):
    def __init__(self, w=None, size=10000, dim=50, initializer=default_initializer, name=""):
        if w is None:
            self.size = size
            self.dim = dim
            self.W = get_variable((size, dim), initializer=initializer, name=name)
        else:
            self.size = w.shape[0]
            self.dim = w.shape[1]
            self.W = tf.Variable(w, name=name, dtype=data_type())
        self.l1_norm = tf.reduce_sum(tf.abs(self.W))
        self.l2_norm = tf.reduce_sum(tf.square(self.W))

    def __getitem__(self, item):
        return self.W[item]

    def get_dim(self):
        return self.dim

    def get_value(self, sess):
        return sess.run(self.W)


class WordEmbedding(Embedding):
    def __init__(self, word_idx, dim=50, initializer=default_initializer, name="", vervose=True):
        self.n_words = len(word_idx)
        self.dim = dim
        self.initializer = initializer
        self.word_idx = word_idx
        self.idx_word = {idx: word for word, idx in self.word_idx.items()}
        super(WordEmbedding, self).__init__(size=self.n_words, dim=self.dim, initializer=initializer, name=name)
        if vervose:
            logging.info("Word embeddings finished initialization.\n")
            logging.info("Word size: %d\n" % self.n_words)
            logging.info("Word dimension: %d\n" % self.dim)






