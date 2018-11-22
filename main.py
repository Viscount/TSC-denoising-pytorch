#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import csv
import pickle

import model.skip_gram as skip_gram
import model.tsc_embed as tsc_embed
import model.e2eshallow as e2e

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def preprocess(output_file):
    # load data from csv
    # seasons = pd.read_csv("./data/bangumi.csv", delimiter=",", encoding="utf-8")
    # episodes = pd.read_csv("./data/episode.csv", delimiter=",", encoding="utf-8")
    danmaku_complete = pd.read_csv("./data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
                                   quoting=csv.QUOTE_NONE, low_memory=False)
    danmaku_complete = danmaku_complete.fillna(-1)
    # skip-gram model
    # dm_set = skip_gram.build_dataset(danmaku_complete)
    # dm_set = tsc_embed.build_dataset(danmaku_complete)
    dm_train_set, dm_test_set = e2e.build_dataset(danmaku_complete)

    # pickle.dump(dm_set, open(output_file, 'wb'))
    pickle.dump(dm_train_set, open('./tmp/e2e_train_dataset.pkl', 'wb'))
    pickle.dump(dm_test_set, open('./tmp/e2e_test_dataset.pkl', 'wb'))


if __name__ == "__main__":
    dataset_file = './tmp/tsc_dataset.pkl'
    # preprocess(dataset_file)

    # build datasettrain_set = pickle.load(open('./tmp/e2e_train_dataset.pkl', 'wb'))
    # dataset = pickle.load(open(dataset_file, 'rb'))
    # print(type(dataset))
    train_set = pickle.load(open('./tmp/e2e_train_dataset.pkl', 'rb'))
    test_set = pickle.load(open('./tmp/e2e_test_dataset.pkl', 'rb'))

    # train
    # skip_gram.train(dataset)
    # tsc_embed.train(dataset)
    e2e.train(train_set, test_set)
