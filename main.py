#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import csv
import pickle

import model.skip_gram as skip_gram
import model.tsc_embed as tsc_embed

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def preprocess(output_file):
    # load data from csv
    seasons = pd.read_csv("./data/bangumi.csv", delimiter=",", encoding="utf-8")
    episodes = pd.read_csv("./data/episode.csv", delimiter=",", encoding="utf-8")
    danmaku_complete = pd.read_csv("./data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
                                   quoting=csv.QUOTE_NONE, low_memory=False)
    danmaku_complete = danmaku_complete.fillna(-1)
    # skip-gram model
    # dm_set = skip_gram.build_dataset(danmaku_complete)
    dm_set = tsc_embed.build_dataset(danmaku_complete)

    pickle.dump(dm_set, open(output_file, 'wb'))


if __name__ == "__main__":
    dataset_file = './tmp/tsc_dataset.pkl'
    preprocess(dataset_file)

    # build dataset
    dataset = pickle.load(open(dataset_file, 'rb'))
    print(type(dataset))

    # train
    # skip_gram.train(dataset)
    tsc_embed.train(dataset)
