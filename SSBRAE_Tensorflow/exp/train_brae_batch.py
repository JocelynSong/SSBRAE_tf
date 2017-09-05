import argparse
import sys
import os
import logging
import time

import tensorflow as tf
import numpy as np

from src.utils import get_train_sequence, pre_logger
from src.embedding import WordEmbedding
from src.data_utils.phrase_utils import prepare_data
from src.config import BRAEConfig
from src.rae_batch import BilingualPhraseRAE

FLAGS = None
logger = logging.getLogger(__name__)


def main(_):
    print("src para file:%s\n" % FLAGS.src_para)
    phrase_file = FLAGS.phrase_file
    src_para = FLAGS.src_para
    tar_para = FLAGS.tar_para
    trans_file = FLAGS.trans_file
    src_phrase_list, tar_phrase_list, bi_phrase_list, src_word_idx, tar_word_idx = prepare_data(phrase_file,
                                                                                                src_para,
                                                                                                tar_para,
                                                                                                trans_file)
    brae_config = BRAEConfig(FLAGS.config_name)
    src_word_embedding = WordEmbedding(src_word_idx, dim=50, name="src_word_embedding")
    tar_word_embedding = WordEmbedding(tar_word_idx, dim=50, name="tar_word_embedding")
    sess = tf.Session()

    brae_encoder = BilingualPhraseRAE(src_word_embedding, tar_word_embedding, brae_config.activation,
                                      brae_config.normalize, brae_config.weight_rec, brae_config.weight_sem,
                                      brae_config.weight_embedding, brae_config.alpha, brae_config.max_src_len,
                                      brae_config.max_tar_len, brae_config.n_epoch, brae_config.batch_size,
                                      brae_config.dropout, brae_config.optimizer_config, sess)
    train_phrase_list = bi_phrase_list[: -2 * brae_config.batch_size]
    valid_phrase_list = bi_phrase_list[-2 * brae_config.batch_size: -brae_config.batch_size]
    test_phrase_list = bi_phrase_list[-brae_config.batch_size:]

    pre_logger("brae")
    logger.info("Now train brae encoder\n")
    for i in range(brae_encoder.n_epoch):
        logger.info("Now train brae encoder epoch %d\n" % i)
        losses = []
        start_time = time.time()

        train_phrase_index = get_train_sequence(train_phrase_list, brae_encoder.batch_size)
        num_batches = int(len(train_phrase_index) / brae_encoder.batch_size)
        for j in range(num_batches):
            src_pos, tar_pos, src_neg, tar_neg = brae_encoder.get_batch(src_phrase_list, tar_phrase_list,
                                                                        train_phrase_list, train_phrase_index,
                                                                        src_word_idx, tar_word_idx, j)
            loss = brae_encoder.train_step(src_pos, tar_pos, src_neg, tar_neg)
            logger.info("train brae encoder epoch %d, step %d, train loss: %f\n" % (i, j, loss))
            losses.append(loss)

        valid_phrase_index = get_train_sequence(valid_phrase_list, brae_encoder.batch_size)
        num_batches = int(len(valid_phrase_index) / brae_encoder.batch_size)
        dev_loss = []
        for j in range(num_batches):
            src_pos, tar_pos, src_neg, tar_neg = brae_encoder.get_batch(src_phrase_list, tar_phrase_list,
                                                                        valid_phrase_list, valid_phrase_index,
                                                                        src_word_idx, tar_word_idx, j)
            dev_loss.append(brae_encoder.predict_step(src_pos, tar_pos, src_neg, tar_neg))

        use_time = time.time() - start_time
        logger.info("train brae encoder epoch %d, use time:%d, train loss:%f, development loss: %f\n"
                    % (i, use_time, sess.run(tf.reduce_mean(losses)), sess.run(tf.reduce_mean(losses))))

        checkpoint_path = os.path.join(FLAGS.train_dir, "brae_encoder.epoch%d.ckpt" % i)
        #brae_encoder.saver.save(brae_encoder.sess, checkpoint_path, global_step=brae_encoder.global_step)
        brae_encoder.saver.save(brae_encoder.sess, checkpoint_path)

    test_phrase_index = get_train_sequence(test_phrase_list, brae_encoder.batch_size)
    num_batches = int(len(test_phrase_index) / brae_encoder.batch_size)
    test_loss = []
    for j in range(num_batches):
        src_pos, tar_pos, src_neg, tar_neg = brae_encoder.get_batch(src_phrase_list, tar_phrase_list,
                                                                    test_phrase_list, test_phrase_index,
                                                                    src_word_idx, tar_word_idx, j)
        test_loss.append(brae_encoder.predict_step(src_pos, tar_pos, src_neg, tar_neg))
    logger.info("train brae encoder, test loss is: %f\n" % sess.run(tf.reduce_mean(test_loss)))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--phrase_file", type=str, default="..\\data\\test\\test_phrase", help="phrase file name")
    parse.add_argument("--src_para", type=str, default="..\\data\\test\\test_para.zh",
                       help="source paraphrase file name")
    parse.add_argument("--tar_para", type=str, default="..\\data\\test\\test_para.en",
                       help="target paraphrase file name")
    parse.add_argument("--trans_file", type=str, default="..\\data\\test\\test_phrase", help="translation file name")
    parse.add_argument("--config_name", type=str, default="..\\conf\\brae_config.conf",
                       help="brae configuration file")
    parse.add_argument("--train_dir", type=str, default="..\\exp\\model\\brae", help="brae training directory")

    FLAGS, unparsed = parse.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








