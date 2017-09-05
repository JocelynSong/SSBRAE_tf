import tensorflow as tf

__author__ = "Jocelyn"


def identity(x):
    return x


class Activation(object):
    def __init__(self, method):
        self.method = method
        method_name = method.lower()
        if method_name == "sigmoid":
            self.func = tf.nn.sigmoid
        elif method_name == "tanh":
            self.func = tf.nn.tanh
        elif method_name == "relu":
            self.func = tf.nn.relu
        elif method_name == "elu":
            self.func = tf.nn.elu
        elif method_name == "identity":
            self.func = identity
        else:
            raise ValueError("No such activation method:%s" % method_name)

    def activate(self, x):
        return self.func(x)


