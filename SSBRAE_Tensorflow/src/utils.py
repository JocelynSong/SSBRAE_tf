import tensorflow as tf
import numpy as np
import logging

from src import default_initializer
from src.Initializer import data_type


def get_variable(shape, name, initializer=default_initializer):
    ini = initializer.generate(shape=shape)
    return tf.Variable(ini, name=name, dtype=data_type())


def ndarray_slice(x, n, dim):
    if x.ndim == 1:
        return x[n * dim: (n + 1) * dim]
    elif x.dim == 2:
        return x[:, n * dim: (n + 1) * dim]
    elif x.ndim == 3:
        return x[:, :, n * dim: (n + 1) * dim]
    else:
        raise ValueError("Such type of x is not valid!")


def array2str(array, space=' '):
    return space.join(['%.6f' % b for b in array])


def align_batch_size(train_index, batch_size):
    if len(train_index) % batch_size == 0:
        return train_index
    else:
        remain = batch_size - len(train_index) % batch_size
        for i in range(remain):
            ind = np.random.randint(0, len(train_index)-1)
            train_index.append(train_index[ind])
        return train_index


def get_train_sequence(train_x, batch_size):
    train_index = list(range(len(train_x)))
    train_index = align_batch_size(train_index, batch_size)
    np.random.shuffle(train_index)
    return train_index


def pre_logger(log_file_name, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    # Logging configuration
    # Set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger()
    init_logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("log/{}.log".format(log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_handler_level)
    # Screen logger
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)
    init_logger.addHandler(file_handler)
    init_logger.addHandler(screen_handler)
    return init_logger
