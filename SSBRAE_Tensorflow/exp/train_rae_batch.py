import argparse

import tensorflow as tf
import numpy as np

import os
import sys
import time
import logging


from src.data_utils.phrase_utils import prepare_data
from src.config import BRAEConfig
from src.rae_batch import RAEEncoder
from src.embedding import WordEmbedding
from src.utils import get_train_sequence

FLAGS = None
logger = logging.getLogger(__name__)


def main(_):
    phrase_file = FLAGS.phrase_file
    src_para_file = FLAGS.src_para_file
    tar_para_file = FLAGS.tar_para_file
    trans_file = FLAGS.trans_file
    src_phrase_list, tar_phrase_list, bi_phrase_list, src_word_idx, tar_word_idx = prepare_data(phrase_file,
                                                                                                src_para_file,
                                                                                                tar_para_file,
                                                                                                trans_file)
    # src rae encoder
    src_config_name = FLAGS.src_config_name
    src_rae_config = BRAEConfig(src_config_name)
    src_embedding = WordEmbedding(src_word_idx, dim=50, name="src_embedding")
    sess = tf.Session()
    src_rae_encoder = RAEEncoder(src_rae_config.activation, src_embedding, src_rae_config.normalize,
                                 src_rae_config.weight_rec, src_rae_config.weight_embedding, src_rae_config.n_epoch,
                                 src_rae_config.max_src_len, src_rae_config.batch_size, src_rae_config.dropout,
                                 src_rae_config.optimizer_config, sess, name="rae_encoder")
    # tar rae encoder
    tar_config_name = FLAGS.tar_config_name
    tar_rae_config = BRAEConfig(tar_config_name)
    tar_embedding = WordEmbedding(tar_word_idx, dim=50, name="tar_embedding")
    tar_rae_encoder = RAEEncoder(tar_rae_config.activation, tar_embedding, tar_rae_config.normalize,
                                 tar_rae_config.weight_rec, tar_rae_config.weight_embedding, tar_rae_config.n_epoch,
                                 tar_rae_config.max_tar_len, tar_rae_config.batch_size, tar_rae_config.dropout,
                                 tar_rae_config.optimizer_config, sess, name="tar_rae_encoder")

    train_phrase_list = src_phrase_list[: - 2 * src_rae_config.batch_size],
    valid_phrase_list = src_phrase_list[-2 * src_rae_config.batch_size: - src_rae_config.batch_size]
    test_phrase_list = src_phrase_list[- src_rae_config.batch_size:]

    logger.info("Now train the src rae encoder:\n")
    for i in range(src_rae_encoder.n_epoch):
        logger.info("Now train src rae epoch %d\n" % (i + 1))
        start_time = time.time()

        src_train_index = get_train_sequence(train_phrase_list, src_rae_encoder.batch_size)
        batch_number = int(len(src_train_index) / src_rae_encoder.batch_size)
        losses = []
        for j in range(batch_number):
            inputs = src_rae_encoder.get_batch(train_phrase_list, src_train_index, j, src_word_idx)
            loss = src_rae_encoder.train_step(inputs)
            logging.info("src rae epoch %d, step %d, loss: %f\n" % (i, j, loss))
            losses.append(loss)

        src_valid_index = get_train_sequence(valid_phrase_list, src_rae_encoder.batch_size)
        valid_batches = int(len(src_valid_index) / src_rae_encoder.batch_size)
        dev_loss = []
        for j in range(valid_batches):
            inputs = src_rae_encoder.get_batch(valid_phrase_list, src_valid_index, j, src_word_idx)
            dev_loss.append(src_rae_encoder.predict_step(inputs))

        use_time = time.time() - start_time
        logger.info("src rae epoch %d, time: %d, train loss:%f, development loss:%f\n"
                    % (i, use_time, sess.run(tf.reduce_mean(losses)), sess.run(tf.reduce_mean(dev_loss))))

        checkpoint_path = os.path.join(FLAGS.train_dir, "src_rae.epoch%d.ckpt" % i)
        src_rae_encoder.saver.save(src_rae_encoder.sess, checkpoint_path, global_step=src_rae_encoder.global_step)

    src_test_index = get_train_sequence(test_phrase_list, src_rae_encoder.batch_size)
    test_batches = int(len(src_test_index) / src_rae_encoder.batch_size)
    test_loss = []
    for j in range(test_batches):
        inputs = src_rae_encoder.get_batch(test_phrase_list, src_test_index, j, src_word_idx)
        test_loss.append(src_rae_encoder.predict_step(inputs))
    logger.info("src test loss : %f\n" % sess.run(tf.reduce_mean(test_loss)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--phrase_file", default="/data/phrase_file", help="file name for phrase")
    parser.add_argument("--src_para_file", default="/data/src_para_file", help="source paraphrase file name")
    parser.add_argument("--tar_para_file", default="/data/tar_para_file", help="target paraphrase file name")
    parser.add_argument("--trans_file", default="/data/trans_file", help="translation file name")
    parser.add_argument("--src_config_name", default="/conf/src.config", help="source configuration file name")
    parser.add_argument("--tar_config_name", default="/conf/tar.config", help="target configuration file name")
    parser.add_argument("--train_dir", default="/exp/model", help="training directory")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)













