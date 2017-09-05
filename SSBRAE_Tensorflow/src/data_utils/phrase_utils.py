# -*- coding: utf-8 -*-

from collections import OrderedDict
from src import OOV_KEY
from src.data_utils import para_weight_threshold

WORD_INDEX = 0
TEXT_INDEX = 1
PARA_INDEX = 2
TRAN_INDEX = 3


def phrase_list_generator(filename, src_max=7, tar_max=7):
    """
    read phrases from file
    :param filename:
    :param src_max:
    :param tar_max:
    :return:
    """
    f = open(filename, "r", encoding="utf-8")
    for line in f:
        s = line.strip().split("|||")
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weights = s[2].strip().split()
        weight1, weight2 = int(weights[0].strip()), int(weights[0].strip())
        yield src_phrase, tar_phrase, weight1, weight2


def para_list_generator(filename, src_max=7, tar_max=7):
    """
    read paraphrase pairs from file
    :param filename:
    :param src_max:
    :param tar_max:
    :return:
    """
    f = open(filename, "r", encoding="utf-8")
    for line in f:
        s = line.strip().split("|||")
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weight1, weight2 = float(s[2].strip()), float(s[3].strip())
        if weight1 < para_weight_threshold or weight2 < para_weight_threshold:
            continue
        yield src_phrase, tar_phrase, weight1, weight2


def trans_list_generator(filename, src_max=7, tar_max=7):
    """
    read translation pairs from file
    :param filename:
    :param src_max:
    :param tar_max:
    :return:
    """
    f = open(filename, "r", encoding="utf-8")
    for line in f:
        s = line.strip().split("|||")
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weights = s[2].strip().split()
        weight1, weight2 = int(weights[0].strip()), int(weights[0].strip())
        yield src_phrase, tar_phrase, weight1, weight2


def add_word(vocab, word):
    """
    add word to vocab
    :param vocab:
    :param word:
    :return:
    """
    if word not in vocab:
        vocab[word] = {"index": len(vocab), "count": 1}
    else:
        vocab[word]["count"] += 1


def words2index(vocab, words):
    """
    continuing words(phrase) to word index
    :param vocab:
    :param words:
    :return:
    """
    return [vocab[word] if word in vocab else vocab[OOV_KEY] for word in words.strip().split()]


def read_phrase_pair_vocab(filename):
    """
    reading words from phrase pair file
    :param filename:
    :return:
    """
    src_vocab = {}
    tar_vocab = {}
    for src, tar,  _, _ in phrase_list_generator(filename):
        for word in src.split():
            add_word(src_vocab, word)
        for word in tar.split():
            add_word(tar_vocab, word)
    return src_vocab, tar_vocab


def add_para_word_vocab(filename, word_idx):
    """
    reading words from paraphrase file
    :param filename:
    :param word_idx:
    :return:
    """
    for src, tar, _, _ in para_list_generator(filename):
        for word in src.split():
            add_word(word_idx, word)
        for word in tar.split():
            add_word(word_idx, word)
    return word_idx


def get_phrase_instance(word_idx, phrase):
    """
    get a instance for a phrase
    :param word_idx:
    :param phrase:
    :return:
    """
    return [words2index(word_idx, phrase), phrase, list(), dict()]


def add_trans_word_vocab(filename, src_word_idx, tar_word_idx):
    """
    add word of translation file to vocab
    :param filename:
    :param src_word_idx:
    :param tar_word_idx:
    :return:
    """
    for src, tar, _, _, in trans_list_generator(filename):
        for word in src.split():
            add_word(src_word_idx, word)
        for word in tar.split():
            add_word(tar_word_idx, word)
    return src_word_idx, tar_word_idx


def filter_vocab(word_idx, min_count=5):
    filter_dict = {}
    for word, value in word_idx.items():
        if value["count"] >= min_count:
            filter_dict[word] = len(filter_dict)
    filter_dict[OOV_KEY] = len(filter_dict)
    return filter_dict


def read_phrase_list(phrase_file, src_word_idx, tar_word_idx, src_max_len=7, tar_max_len=7):
    src_phrase_idx = dict()
    tar_phrase_idx = dict()
    src_phrase_list = list()
    tar_phrase_list = list()
    bi_phrase_list = list()
    src_index = 0
    tar_index = 0
    for src_phrase, tar_phrase, weight1, weight2 in phrase_list_generator(phrase_file, src_max_len, tar_max_len):
        if src_phrase not in src_phrase_idx:
            src_phrase_idx[src_phrase] = src_index
            src_index += 1
            src_phrase_list.append(get_phrase_instance(src_word_idx, src_phrase))
        if tar_phrase not in tar_phrase_idx:
            tar_phrase_idx[tar_phrase] = tar_index
            tar_index += 1
            tar_phrase_list.append(get_phrase_instance(tar_word_idx, tar_phrase))

        src_idx = src_phrase_idx[src_phrase]
        tar_idx = tar_phrase_idx[tar_phrase]
        src_phrase_list[src_idx][TRAN_INDEX][tar_idx] = float(weight1)
        tar_phrase_list[tar_idx][TRAN_INDEX][src_idx] = float(weight2)
        bi_phrase_list.append((src_idx, tar_idx))
    return src_phrase_list, tar_phrase_list, bi_phrase_list


def read_para_list(para_file, phrase_list, word_idx):
    text2pid = dict()
    for phrase, i in zip(phrase_list, range(len(phrase_list))):
        text2pid[phrase[TEXT_INDEX]] = i
    for src_phrase, tar_phrase, weight1, weight2 in para_list_generator(para_file):
        if float(weight1) < para_weight_threshold or float(weight2) < para_weight_threshold:
            continue
        if src_phrase not in text2pid:
            phrase_list.append(get_phrase_instance(word_idx, src_phrase))
            text2pid[src_phrase] = len(phrase_list) - 1
        if tar_phrase not in text2pid:
            phrase_list.append(get_phrase_instance(word_idx, tar_phrase))
            text2pid[tar_phrase] = len(phrase_list) - 1
        src_idx = text2pid[src_phrase]
        tar_idx = text2pid[tar_phrase]
        phrase_list[src_idx][PARA_INDEX].append((tar_idx, float(weight1)))
        phrase_list[tar_idx][PARA_INDEX].append((src_idx, float(weight2)))
    return phrase_list


def read_trans_list(bi_count_file, src_phrase_list, tar_phrase_list, src_word_idx, tar_word_idx):
    src_text2pid = dict()
    tar_text2pid = dict()
    for phrase, i in zip(src_phrase_list, range(len(src_phrase_list))):
        src_text2pid[phrase[TEXT_INDEX]] = i
    for phrase, i in zip(tar_phrase_list, range(len(tar_phrase_list))):
        tar_text2pid[phrase[TEXT_INDEX]] = i

    for src_phrase, tar_phrase, weight1, weight2 in trans_list_generator(bi_count_file):
        if src_phrase not in src_text2pid:
            src_phrase_list.append(get_phrase_instance(src_word_idx, src_phrase))
            src_text2pid[src_phrase] = len(src_phrase_list) - 1
        if tar_phrase not in tar_text2pid:
            tar_phrase_list.append(get_phrase_instance(tar_word_idx, tar_phrase))
            tar_text2pid[tar_phrase] = len(tar_phrase_list) - 1
        src_phrase_idx = src_text2pid[src_phrase]
        tar_phrase_idx = tar_text2pid[tar_phrase]
        src_phrase_list[src_phrase_idx][TRAN_INDEX][tar_phrase_idx] = float(weight1)
        tar_phrase_list[tar_phrase_idx][TRAN_INDEX][src_phrase_idx] = float(weight2)

    def calc_translate_probability(phrase_list):
        for p in phrase_list:
            if len(p[TRAN_INDEX]) == 0:
                continue
            else:
                total_value = sum((p[TRAN_INDEX]).values())
                for key, value in p[TRAN_INDEX].items():
                    p[TRAN_INDEX][key] = float(value)/total_value

    calc_translate_probability(src_phrase_list)
    calc_translate_probability(tar_phrase_list)

    for phrase in src_phrase_list:
        phrase[TRAN_INDEX] = OrderedDict(sorted(phrase[TRAN_INDEX].items(), key=lambda d: d[1], reverse=True))
    for phrase in tar_phrase_list:
        phrase[TRAN_INDEX] = OrderedDict(sorted(phrase[TRAN_INDEX].items(), key=lambda d: d[1], reverse=True))
    return src_phrase_list, tar_phrase_list


def prepare_data(phrase_file, src_para_file, tar_para_file, trans_file):
    src_word_idx, tar_word_idx = read_phrase_pair_vocab(phrase_file)
    src_word_idx = add_para_word_vocab(src_para_file, src_word_idx)
    tar_word_idx = add_para_word_vocab(tar_para_file, tar_word_idx)
    src_word_idx, tar_word_idx = add_trans_word_vocab(trans_file, src_word_idx, tar_word_idx)
    src_word_idx = filter_vocab(src_word_idx)
    tar_word_idx = filter_vocab(tar_word_idx)
    src_phrase_list, tar_phrase_list, bi_phrase_list = read_phrase_list(phrase_file, src_word_idx,
                                                                        tar_word_idx)
    src_phrase_list = read_para_list(src_para_file, src_phrase_list, src_word_idx)
    tar_phrase_list = read_para_list(tar_para_file, tar_phrase_list, tar_word_idx)
    src_phrase_list, tar_phrase_list = read_trans_list(trans_file, src_phrase_list, tar_phrase_list,
                                                       src_word_idx, tar_word_idx)
    return src_phrase_list, tar_phrase_list, bi_phrase_list, src_word_idx, tar_word_idx


def clean_text(phrase_list):
    import gc
    for phrase in phrase_list:
        phrase[TEXT_INDEX] = None
    gc.collect()
    gc.collect()
    gc.collect()
    return phrase_list


def clean_trans(phrase_list):
    import gc
    for phrase in phrase_list:
        phrase[TRAN_INDEX] = None
    gc.collect()
    gc.collect()
    gc.collect()
    return phrase_list


