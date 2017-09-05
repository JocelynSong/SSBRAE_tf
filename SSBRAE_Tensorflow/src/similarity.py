import tensorflow as tf

__author__ = "Jocelyn"


def cosine_similarity_batch(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    norm_x = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    norm_y = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
    cos = tf.reduce_sum(x * y, axis=1) / (norm_x * norm_y)
    return cos


def cosine_similarity(x, y):
    norm_x = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    norm_y = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
    return tf.reduce_sum(x * y) / (norm_x * norm_y)
