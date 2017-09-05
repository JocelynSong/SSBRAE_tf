from abc import ABCMeta, abstractmethod
from six import add_metaclass
import tensorflow as tf
import numpy as np

__author__ = "Jocelyn"

use_fp = False


def data_type():
    if use_fp:
        return tf.float16
    else:
        return tf.float32


@add_metaclass(ABCMeta)
class Initializer(object):
    @abstractmethod
    def generate(self, shape):
        """
        Generate a initial set for parameters from a given distribution
        :param shape:
        :return:
        """


class NormalizeInitializer(Initializer):
    def __init__(self, mean, std):
        """

        :param mean: mean center of the distribution
        :param std: standard deviation of the distribution
        """
        self.mean = mean
        self.std = std

    def __str__(self):
        return "Normal distribution: (mean=%f std=%f\n) " % (self.mean, self.std)

    def generate(self, shape):
        """

        :param shape:
        :return:
        """
        m = np.random.normal(loc=self.mean, scale=self.std, size=shape)
        return as_float(m)


class UniformInitializer(Initializer):
    def __init__(self, scale):
        self.scale = scale

    def __str__(self):
        return "uniform distribution: scale=%f\n" % self.scale

    def generate(self, shape):
        """
        :param shape:
        :return:
        """
        m = np.random.uniform(low=-self.scale, high=self.scale, size=shape)
        return as_float(m)


class GlorotUniformInitializer(Initializer):
    def __str__(self):
        return "glorot uniform distribution"

    def generate(self, shape):
        fan_in = shape[0]
        fan_out = shape[1]
        scale = np.sqrt(6. / (fan_in + fan_out))
        m = np.random.uniform(low=-scale, high=scale, size=shape)
        return as_float(m)


class IdentityInitializer(Initializer):
    def __init__(self, mat):
        self.matrix = mat

    def __str__(self):
        return "Identity initializer"

    def generate(self, shape):
        if len(shape) != 2:
            raise ValueError("The length of shape must be 2!")
        row, column = shape
        return self.matrix * tf.eye(row, column)


class OrthogonalInitializer(Initializer):
    def __init__(self, gain):
        if gain == "relu":
            gain = np.sqrt(2)
        self.gain = gain

    def __str__(self):
        return "orthogonal distribution"

    def generate(self, shape):
        if len(shape) < 2:
            raise ValueError("The length of shape must be >= 2!")
        flat_shape = shape[0], np.prod(shape[1:])
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return as_float(q)


def as_float(variable):
    return tf.cast(variable, data_type())


