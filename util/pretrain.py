#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
import multiprocessing
import pandas as pd
import csv
import os
import numpy as np
from util.word_segment import word_segment
import pickle
import logging

logging.basicConfig(level=logging.INFO)


class TscTaggedDocument(object):
    def __init__(self, danmakus):
        self.danmakus = danmakus

    def __iter__(self):
        for index, row in self.danmakus.iterrows():
            content_words = word_segment(str(row['content']))
            yield content_words


def train(danmakus, model_name):
    tsc_docs = TscTaggedDocument(danmakus)
    model = Word2Vec(size=200, window=5, min_count=2, iter=10, workers=multiprocessing.cpu_count())
    print('Building vocabulary......')
    model.build_vocab(tsc_docs)
    print('Training word2vec model......')
    model.train(tsc_docs, total_examples=model.corpus_count, epochs=model.iter)
    print('Vocabulary size:', len(model.wv.vocab))
    model.save("../tmp/" + model_name)
    return


def get_weight(model, dictionary, dim):
    words_count = len(dictionary)
    print("%d words loaded in dictionary." % words_count)
    result = np.zeros((words_count, dim))
    count = 0
    for word in dictionary:
        idx = dictionary[word]
        if word == 'UNK':
            unk_id = idx
        elif word in model.wv.vocab:
            result[idx] = model.wv[word]
            count += 1
        else:
            continue
    print("%d words transfered to matrix." % count)
    return result


if __name__ == "__main__":
    season_id = '24581'

    # danmaku_complete = pd.read_csv("../data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
    #                                quoting=csv.QUOTE_NONE, low_memory=False)
    # danmaku_complete = danmaku_complete.fillna(-1)
    # danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    # train(danmaku_selected, "24581_dm_word_embedding.model")

    word_model = Word2Vec.load(os.path.join('../tmp', season_id, 'py_word_embedding.model'))
    word_dim = 200
    print(len(word_model.wv.vocab))

    dataset_type = 'triplet'

    dataset = pickle.load(open(os.path.join('../tmp', season_id, dataset_type+'_train_dataset.pkl'), 'rb'))
    dictionary = dataset.py_word_to_ix
    weight = get_weight(word_model, dictionary, word_dim)
    np.savetxt(os.path.join('../tmp', season_id, 'py_weights.txt'), weight)
