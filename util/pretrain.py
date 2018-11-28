#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
import multiprocessing
import pandas as pd
import csv
import numpy as np
from util.word_segment import word_segment
import pickle


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
    # danmaku_complete = pd.read_csv("../data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
    #                                quoting=csv.QUOTE_NONE, low_memory=False)
    # danmaku_complete = danmaku_complete.fillna(-1)
    # danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    # train(danmaku_selected, "dm_word_embedding_200.model")

    word_model = Word2Vec.load("../tmp/dm_word_embedding_200.model")
    word_dim = 200
    print(len(word_model.wv.vocab))

    dataset = pickle.load(open('../tmp/e2e_we_train_dataset.pkl', 'rb'))
    dictionary = dataset.word_to_ix
    weight = get_weight(word_model, dictionary, word_dim)
    np.savetxt("../tmp/weights.txt", weight)