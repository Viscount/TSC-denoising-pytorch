#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import csv
import pickle

import model.skip_gram as skip_gram


if __name__ == "__main__":
    # load data from csv
    seasons = pd.read_csv("./data/bangumi.csv", delimiter=",", encoding="utf-8")
    episodes = pd.read_csv("./data/episode.csv", delimiter=",", encoding="utf-8")
    danmaku_complete = pd.read_csv("./data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
                                   quoting=csv.QUOTE_NONE, low_memory=False)
    danmaku_complete = danmaku_complete.fillna(-1)

    # build dataset
    dm_set = skip_gram.build_dataset(danmaku_complete)

    # train
    skip_gram.train(dm_set)
