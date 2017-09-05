import argparse
import sys
import os
import time
import logging

import tensorflow as tf
import numpy as np

from src.embedding import WordEmbedding
from src.utils import get_train_sequence, pre_logger
from src.data_utils.phrase_utils import prepare_data
from src.config import SSBRAEConfig
from src.rae_batch import SSBRAEEncoder

FLAGS = None
logger = logging.getLogger(__name__)


def main(_):
    phrase_file = FLAGS.phrase_file
    src_para_file = FLAGS.src_para
    tar_para_file = FLAGS.tar_para
    trans_file = FLAGS.trans_file
    src_phrase_list, tar_phrase_list, bi_phrase_list, src_word_idx, tar_word_idx = prepare_data(phrase_file,
                                                                                                src_para_file,
                                                                                                tar_para_file,
                                                                                                trans_file)
    ssbrae_config = SSBRAEConfig(FLAGS.config_name)
    src_word_embedding = WordEmbedding(src_word_idx, dim=50, name="src_word_embedding")
    tar_word_embedding = WordEmbedding(tar_word_idx, dim=50, name="tar_word_embedding")
    sess = tf.Session()

    ssbrae_encoder = SSBRAEEncoder(src_word_embedding, tar_word_embedding, ssbrae_config.activation,
                                   ssbrae_config.normalize, ssbrae_config.weight_rec, ssbrae_config.weight_sem,
                                   ssbrae_config.weight_embedding, ssbrae_config.alpha, ssbrae_config.beta,
                                   ssbrae_config.max_src_len, ssbrae_config.max_tar_len, ssbrae_config.n_epoch,
                                   ssbrae_config.batch_size, ssbrae_config.dropout, ssbrae_config.optimizer_config,
                                   ssbrae_config.para, ssbrae_config.trans, ssbrae_config.para_num,
                                   ssbrae_config.trans_num, sess)

    train_phrase_list = bi_phrase_list[: -2 * ssbrae_encoder.batch_size]
    valid_phrase_list = bi_phrase_list[-2 * ssbrae_encoder.batch_size: -ssbrae_encoder.batch_size]
    test_phrase_list = bi_phrase_list[-ssbrae_encoder.batch_size:]

    pre_logger("ssbrae")
    logger.info("Now train ssbrae encoder\n")
    for i in range(ssbrae_encoder.n_epoch):
        logger.info("Now train ssbrae encoder epoch %d\n" % i)
        start_time = time.time()
        losses = []

        train_phrase_index = get_train_sequence(train_phrase_list, ssbrae_encoder.batch_size)
        num_batches = int(len(train_phrase_index) / ssbrae_encoder.batch_size)
        for j in range(num_batches):
            (src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para, src_para_weight, tar_para_weight, src_tar_trans,\
             tar_src_trans, src_tar_trans_weight, tar_src_trans_weight) = ssbrae_encoder.get_batch(src_phrase_list,
                                                                                                  tar_phrase_list,
                                                                                                  train_phrase_list,
                                                                                                  train_phrase_index,
                                                                                                  src_word_idx,
                                                                                                  tar_word_idx, j)
            result = ssbrae_encoder.ssbrae_train_step(src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para,
                                                      src_para_weight, tar_para_weight, src_tar_trans,
                                                      tar_src_trans, src_tar_trans_weight, tar_src_trans_weight)
            if ssbrae_encoder.para and ssbrae_encoder.trans:
                logger.info("train ssbrae_para epoch %d, step %d, total loss:%f, loss_l2: %f, loss_rec: %f,"
                            "loss_sem:%f, loss_para:%f, loss_trans:%f\n" % (i, j, result[1], result[2],
                                                                            result[3], result[4],
                                                                            result[5], result[6]))
            elif ssbrae_encoder.para and not ssbrae_encoder.trans:
                logger.info("train ssbrae_para epoch %d, step %d, total loss:%f, loss_l2: %f, loss_rec: %f,"
                            "loss_sem:%f, loss_para:%f\n" % (i, j, result[1], result[2], result[3],
                                                             result[4], result[5]))
            elif ssbrae_encoder.trans and not ssbrae_encoder.para:
                logger.info("train ssbrae_para epoch %d, step %d, total loss:%f, loss_l2: %f, loss_rec: %f,"
                            "loss_sem:%f, loss_trans:%f\n" % (i, j, result[1], result[2], result[3],
                                                              result[4], result[5]))
            else:
                raise ValueError("No such configuration")
            losses.append(result[1:])

        use_time = time.time() - start_time

        valid_phrase_index = get_train_sequence(valid_phrase_list, ssbrae_encoder.batch_size)
        num_batches = int(len(valid_phrase_index) / ssbrae_encoder.batch_size)
        dev_loss = []
        for j in range(num_batches):
            (src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para, src_para_weight, tar_para_weight, src_tar_trans, \
             tar_src_trans, src_tar_trans_weight, tar_src_trans_weight) = ssbrae_encoder.get_batch(src_phrase_list,
                                                                                                   tar_phrase_list,
                                                                                                   valid_phrase_list,
                                                                                                   valid_phrase_index,
                                                                                                   src_word_idx,
                                                                                                   tar_word_idx, j)
            dev_loss.append(ssbrae_encoder.ssbrae_predict_step(src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para,
                                                               src_para_weight, tar_para_weight, src_tar_trans,
                                                               tar_src_trans, src_tar_trans_weight, tar_src_trans_weight))
        logger.info("train ssbrae encoder epoch %d, use time:%d\n" % (i, use_time))
        ave_train_loss = np.average(losses, axis=0)
        ave_dev_loss = np.average(dev_loss, axis=0)
        if ssbrae_encoder.para and ssbrae_encoder.trans:
            logger.info("train: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f, trans loss:%f\n"
                        % (ave_train_loss[0], ave_train_loss[1], ave_train_loss[2], ave_train_loss[3],
                           ave_train_loss[4], ave_train_loss[5]))
            logger.info("dev: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f, trans loss:%f"
                        % (ave_dev_loss[0], ave_dev_loss[1], ave_dev_loss[2], ave_dev_loss[3], ave_dev_loss[4],
                           ave_dev_loss[5]))
        elif ssbrae_encoder.para and not ssbrae_encoder.trans:
            logger.info("train: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f\n"
                        % (ave_train_loss[1], ave_train_loss[2], ave_train_loss[3], ave_train_loss[4],
                           ave_train_loss[5]))
            logger.info("dev: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f"
                        % (ave_dev_loss[0], ave_dev_loss[1], ave_dev_loss[2], ave_dev_loss[3], ave_dev_loss[4]))
        elif ssbrae_encoder.trans and not ssbrae_encoder.para:
            logger.info("train: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, trans loss:%f\n"
                        % (ave_train_loss[1], ave_train_loss[2], ave_train_loss[3], ave_train_loss[4],
                           ave_train_loss[5]))
            logger.info("dev: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, trans loss:%f"
                        % (ave_dev_loss[0], ave_dev_loss[1], ave_dev_loss[2], ave_dev_loss[3], ave_dev_loss[4]))

        checkpoint_path = os.path.join(FLAGS.train_dir, "ssbare_encoder.epoch%d.ckpt" % i)
        #ssbrae_encoder.saver.save(ssbrae_encoder.sess, checkpoint_path, global_step=ssbrae_encoder.global_step)
        ssbrae_encoder.saver.save(ssbrae_encoder.sess, checkpoint_path)

    test_phrase_index = get_train_sequence(test_phrase_list, ssbrae_encoder.batch_size)
    num_batches = int(len(test_phrase_index) / ssbrae_encoder.batch_size)
    test_loss = []
    for j in range(num_batches):
        (src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para, src_para_weight, tar_para_weight, src_tar_trans, \
         tar_src_trans, src_tar_trans_weight, tar_src_trans_weight) = ssbrae_encoder.get_batch(src_phrase_list,
                                                                                               tar_phrase_list,
                                                                                               test_phrase_list,
                                                                                               test_phrase_index,
                                                                                               src_word_idx,
                                                                                               tar_word_idx, j)
        test_loss.append(ssbrae_encoder.ssbrae_predict_step(src_pos, tar_pos, src_neg, tar_neg, src_para, tar_para,
                                                            src_para_weight, tar_para_weight, src_tar_trans,
                                                            tar_src_trans, src_tar_trans_weight, tar_src_trans_weight))

    ave_test_loss = np.average(test_loss, axis=0)
    if ssbrae_encoder.para and ssbrae_encoder.trans:
        logger.info("test: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f, trans loss:%f"
                    % (ave_test_loss[0], ave_test_loss[1], ave_test_loss[2], ave_test_loss[3], ave_test_loss[4],
                       ave_test_loss[5]))
    elif ssbrae_encoder.para and not ssbrae_encoder.trans:
        logger.info("test: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, para loss:%f"
                    % (ave_test_loss[0], ave_test_loss[1], ave_test_loss[2], ave_test_loss[3], ave_test_loss[4]))
    elif ssbrae_encoder.trans and not ssbrae_encoder.para:
        logger.info("test: total loss:%f, l2 loss:%f, rec loss:%f, sem loss:%f, trans loss:%f"
                    % (ave_test_loss[0], ave_test_loss[1], ave_test_loss[2], ave_test_loss[3], ave_test_loss[4]))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--phrase_file", type=str, default="..\\data\\test\\test_phrase", help="phrase file name")
    parse.add_argument("--src_para", type=str, default="..\\data\\test\\test_para.zh",
                       help="source paraphrase file name")
    parse.add_argument("--tar_para", type=str, default="..\\data\\test\\test_para.en",
                       help="target paraphrase file name")
    parse.add_argument("--trans_file", type=str, default="..\\data\\test\\test_phrase", help="translation file name")
    parse.add_argument("--config_name", type=str, default="..\\conf\\gbrae.conf", help="brae configuration file")
    parse.add_argument("--train_dir", type=str, default="..\\exp\\model\\ssbrae", help="brae training directory")

    FLAGS, unparsed = parse.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

