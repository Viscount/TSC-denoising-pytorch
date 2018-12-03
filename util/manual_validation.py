#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import pickle
from util.word_segment import word_segment
from model.e2eshallow import DmDataset
from model.e2ewordembed import DmTestDataset


def load_season_danmaku_data(file_name, season_id):
    danmaku_complete = pd.read_csv(file_name, delimiter="\t", encoding="utf-8",
                                   quoting=csv.QUOTE_NONE, low_memory=False)
    danmaku_complete = danmaku_complete.fillna(-1)
    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == season_id]
    return danmaku_selected


def load_validation_data(file_name):
    valid_data = pd.read_csv(file_name, delimiter=",", encoding="utf-8")
    data_dict = dict()
    for index, row in valid_data.iterrows():
        data_dict[row['raw_id']] = row['label']
    return data_dict


def build_valid_set(all_danmakus, select_dict):
    samples = []
    for index, row in all_danmakus.iterrows():
        if row['tsc_raw_id'] in select_dict:
            content = row['content']
            words = word_segment(str(content))
            sample = {
                'raw_id': row['tsc_raw_id'],
                'content': words,
                'playback_time': row['playback_time'],
                'label': select_dict[row['tsc_raw_id']]
            }
            samples.append(sample)

    train_set = pickle.load(open('../tmp/e2e_train_dataset.pkl', 'rb'))
    test_set = DmDataset(samples, 2, train_set.max_len, train_set.word_to_ix)

    # train_set = pickle.load(open('../tmp/e2e_we_train_dataset.pkl', 'rb'))
    # test_set = DmTestDataset(samples, train_set.max_len, train_set.word_to_ix)
    return test_set


if __name__ == '__main__':
    danmaku_selected = load_season_danmaku_data("../data/danmaku_complete.csv", "24581")
    valid_data_dict = load_validation_data("../data/manual_validation.csv")
    valid_set = build_valid_set(danmaku_selected, valid_data_dict)
    print(type(valid_set))
    pickle.dump(valid_set, open('../tmp/e2e_valid_dataset.pkl', 'wb'))
