#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import pickle
import torch.utils.data as data
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


def get_test_detail(dataframe, test_dataset, pred_history=None, history_len=0):
    dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )
    id_array = []
    label_array = []
    for batch_idx, sample_dict in enumerate(dataloader):
        raw_id = sample_dict['raw_id'].numpy()
        labels = sample_dict['label'].numpy()
        id_array.extend(raw_id)
        label_array.extend(labels)
    test_dict = dict()
    for index, row in dataframe.iterrows():
        if row['tsc_raw_id'] in id_array:
            test_dict[row['tsc_raw_id']] = {
                'tsc_raw_id': row['tsc_raw_id'],
                'content': row['content']
            }

    if pred_history is not None:
        if history_len == 0:
            for history in pred_history:
                if len(history) > history_len:
                    history_len = len(history)

    test_list = []
    for index in range(len(id_array)):
        raw_id = id_array[index]
        label = label_array[index]
        data_dict = test_dict[raw_id]
        data_dict['label'] = label
        if pred_history is not None:
            tot_accuracy = 0
            for index in range(len(pred_history)):
                his_key_name = 'history' + str(index)
                acc_key_name = 'accuracy' + str(index)
                history = pred_history[index][raw_id][:history_len]
                data_dict[his_key_name] = history

                hit_count = 0
                for index in range(len(history)):
                    if history[index] == label:
                        hit_count += 1
                data_dict[acc_key_name] = hit_count * 1.0 / len(history)
                tot_accuracy += data_dict[acc_key_name]
            data_dict['tot_accuracy'] = tot_accuracy / len(pred_history)
        test_list.append(data_dict)
    df = pd.DataFrame(test_list)
    df = df.sort_values(by='tot_accuracy')
    df.to_csv('../tmp/pycnn_testset.csv', sep='\t', index=False)
    return


if __name__ == '__main__':
    danmaku_selected = load_season_danmaku_data("../data/danmaku_complete.csv", "24581")
    # valid_data_dict = load_validation_data("../data/manual_validation.csv")
    # valid_set = build_valid_set(danmaku_selected, valid_data_dict)
    # print(type(valid_set))
    # pickle.dump(valid_set, open('../tmp/e2e_valid_dataset.pkl', 'wb'))
    test_dataset = pickle.load(open('../tmp/e2e_pycnn_test_dataset.pkl', 'rb'))
    we_pred_history = pickle.load(open('../tmp/e2e_we_history.pkl', 'rb'))
    cnn_pred_history = pickle.load(open('../tmp/e2e_cnn_history.pkl', 'rb'))
    pycnn_pred_history = pickle.load(open('../tmp/e2e_pycnn_history.pkl', 'rb'))
    pred_history = [we_pred_history, cnn_pred_history, pycnn_pred_history]
    get_test_detail(danmaku_selected, test_dataset, pred_history, history_len=9)
