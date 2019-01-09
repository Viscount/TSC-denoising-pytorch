#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.word_segment import word_segment
from sklearn.model_selection import train_test_split
import pandas as pd

from datasets.dm_unigram_set import DmUnigramDataset
from datasets.dm_triplet_set import DmTripletTrainDataset, DmTripletTestDataset


def dataset_split(danmaku_selected):
    grouped = danmaku_selected.groupby('episode_id')

    samples = []
    pos_label_set = set()
    neg_label_set = set()

    # build all danmaku samples
    for episode_id, group_data in grouped:
        group_data = group_data.sort_values(by='playback_time')
        episode_lvl_samples = []
        print('Processing Episode %d' % episode_id)
        for index, row in group_data.iterrows():
            content = row['content']
            words = word_segment(str(content))
            pys = word_segment(str(content), mode='py')

            if row['block_level'] >= 9:
                label = 0
                pos_label_set.add(row['tsc_raw_id'])
            elif 0 < row['block_level'] <= 2:
                label = 1
                neg_label_set.add(row['tsc_raw_id'])
            else:
                label = -1
            sample = {
                'raw_id': row['tsc_raw_id'],
                'content': words,
                'pinyin': pys,
                'playback_time': row['playback_time'],
                'label': label
            }
            episode_lvl_samples.append(sample)
        samples.append(episode_lvl_samples)

    # train-test split
    pos_train, pos_test = train_test_split(list(pos_label_set), test_size=0.25, shuffle=True, random_state=42)
    neg_tran, neg_test = train_test_split(list(neg_label_set), test_size=0.25, shuffle=True, random_state=42)
    train_select = set()
    train_select.update(pos_train)
    train_select.update(neg_tran)
    test_select = set()
    test_select.update(pos_test)
    test_select.update(neg_test)

    return samples, train_select, test_select


def build(samples, train_select, test_select, dataset_type):
    if dataset_type == 'unigram':
        min_count = 2
        max_len = 0
        train_samples = []
        test_samples = []

        for episode_lvl_samples in samples:
            for sample in episode_lvl_samples:
                if len(sample['content']) > max_len:
                    max_len = len(sample['content'])
                if sample['raw_id'] in train_select:
                    train_samples.append(sample)
                elif sample['raw_id'] in test_select:
                    test_samples.append(sample)

        dm_train_set = DmUnigramDataset(train_samples, min_count, max_len)
        dm_test_set = DmUnigramDataset(test_samples, min_count, max_len, dictionary=dm_train_set.word_to_ix)

        return dm_train_set, dm_test_set

    if dataset_type == 'triplet':
        context_size = 2.5
        common_words_min_count = 3
        min_count = 3
        max_len = 0

        train_samples = []
        test_samples = []
        for episode_lvl_samples in samples:
            episode_lvl_samples_ = []
            for sample in episode_lvl_samples:
                if len(sample['content']) > max_len:
                    max_len = len(sample['content'])
                if sample['raw_id'] in test_select:
                    test_samples.append(sample)
                else:
                    episode_lvl_samples_.append(sample)
            train_samples.append(episode_lvl_samples_)

        dm_train_set = DmTripletTrainDataset(train_samples, min_count, max_len, context_size, common_words_min_count)
        dm_test_set = DmTripletTestDataset(test_samples, max_len, dm_train_set.word_to_ix, dm_train_set.py_word_to_ix)

        return dm_train_set, dm_test_set

    if dataset_type == 'seperate':
        train_samples = []
        test_samples = []
        unlabeled_samples = []

        for episode_lvl_samples in samples:
            for sample in episode_lvl_samples:
                if len(sample['content']) > max_len:
                    max_len = len(sample['content'])
                if sample['raw_id'] in train_select:
                    train_samples.append(sample)
                elif sample['raw_id'] in test_select:
                    test_samples.append(sample)
                else:
                    unlabeled_samples.append(sample)

        return train_samples, test_samples, unlabeled_samples


def dataset_label_fix(dataset, fix_file):
    fix_data = pd.read_csv(fix_file, delimiter=",", encoding="utf-8")
    true_labels = dict()
    for index, row in fix_data.iterrows():
        true_labels[row['raw_id']] = row['label']
    for index in range(len(dataset.samples)):
        raw_id = dataset.samples[index][0]
        if raw_id in true_labels:
            dataset.labels[index] = true_labels[raw_id]
    return dataset


def compare_dataset(dataset, dataset_):
    id_set = set()
    for index in range(len(dataset.samples)):
        raw_id = dataset.samples[index][0]
        id_set.add(raw_id)
    hit_count = 0
    miss_count = 0
    for index in range(len(dataset_.samples)):
        raw_id = dataset.samples[index][0]
        if raw_id in id_set:
            hit_count += 1
        else:
            miss_count += 1
    print("set size: %d, hit count: %d, miss count: %d" % (len(id_set), hit_count, miss_count))
    if miss_count == 0 and hit_count == len(id_set):
        return True
    else:
        return False
