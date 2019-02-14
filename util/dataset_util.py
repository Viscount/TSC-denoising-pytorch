#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.word_segment import word_segment
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import collections
import os

from datasets.dm_unigram_set import DmUnigramDataset
from datasets.dm_triplet_set import DmTripletTrainDataset, DmTripletTestDataset


def build_vocab(words, min_count):
    counter = {'UNK': 0}
    counter.update(collections.Counter(words).most_common())
    rare_words = set()
    for word in counter:
        if word != 'UNK' and counter[word] <= min_count:
            rare_words.add(word)
    for word in rare_words:
        counter['UNK'] += counter[word]
        counter.pop(word)
    print('%d words founded in vocabulary' % len(counter))

    word_to_ix = {
        'EPT': 0
    }
    for word in counter:
        word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def tokenize(word, dictionary):
    if word in dictionary:
        return dictionary[word]
    else:
        return dictionary['UNK']


def dataset_split(danmaku_selected, season_id):
    default_vocab_dictionary_path = os.path.join('./tmp', season_id, 'vocab.dict')
    default_py_dictionary_path = os.path.join('./tmp', season_id, 'pinyin.dict')
    grouped = danmaku_selected.groupby('episode_id')

    samples = []
    vocab_words = []
    pinyins = []
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
            vocab_words.extend(words)
            pinyins.extend(pys)
        samples.append(episode_lvl_samples)

    # build vocab
    min_count = 2
    vocab_dictionary = build_vocab(vocab_words, min_count)
    py_dictionary = build_vocab(pinyins, min_count)
    pickle.dump(vocab_dictionary, open(default_vocab_dictionary_path, 'wb'))
    pickle.dump(py_dictionary, open(default_py_dictionary_path, 'wb'))

    # train-test split
    pos_train, pos_test = train_test_split(list(pos_label_set), test_size=0.25, shuffle=True)
    neg_tran, neg_test = train_test_split(list(neg_label_set), test_size=0.25, shuffle=True)
    train_select = set()
    train_select.update(pos_train)
    train_select.update(neg_tran)
    test_select = set()
    test_select.update(pos_test)
    test_select.update(neg_test)

    return samples, train_select, test_select


def build(season_id, samples, train_select, test_select, dataset_type):
    default_vocab_dictionary_path = os.path.join('./tmp', season_id, 'vocab.dict')
    default_py_dictionary_path = os.path.join('./tmp', season_id, 'pinyin.dict')

    vocab_dictionary = pickle.load(open(default_vocab_dictionary_path, 'rb'))
    py_dictionary = pickle.load(open(default_py_dictionary_path, 'rb'))

    if dataset_type == 'unigram':
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

        dm_train_set = DmUnigramDataset(train_samples, max_len, dictionary=vocab_dictionary)
        dm_test_set = DmUnigramDataset(test_samples, max_len, dictionary=vocab_dictionary)

        return dm_train_set, dm_test_set

    if dataset_type == 'triplet':
        context_size = 2.5
        max_distance = 20
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

        dm_train_set = DmTripletTrainDataset(train_samples, max_len, context_size, max_distance,
                                             dictionarys={'vocab': vocab_dictionary, 'pinyin': py_dictionary})
        dm_test_set = DmTripletTestDataset(test_samples, max_len,
                                           dictionarys={'vocab': vocab_dictionary, 'pinyin': py_dictionary})

        return dm_train_set, dm_test_set

    if dataset_type == 'seperate':
        train_samples = []
        test_samples = []
        unlabeled_samples = []
        max_len = 0

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

    if dataset_type == 'graph':
        # get word features
        word_model = Word2Vec.load(os.path.join('../tmp', season_id, 'dm_word_embedding.model'))
        words_count = len(vocab_dictionary)
        dim = 200
        features = np.zeros((words_count, dim))
        for word in vocab_dictionary:
            idx = vocab_dictionary[word]
            if word == 'UNK':
                unk_id = idx
            elif word in word_model.wv.vocab:
                features[idx] = word_model.wv[word]
            else:
                continue
        # build graph
        adj = np.zeros((words_count, words_count), dtype=np.float32)
        for episode_lvl_samples in samples:
            for sample in episode_lvl_samples:
                content = sample['content']
                for index in range(1, len(content)):
                    start = content[index-1]
                    end = content[index]
                    start_idx = tokenize(start, vocab_dictionary)
                    end_idx = tokenize(end, vocab_dictionary)
                    adj[start_idx][end_idx] = 1

        return features, adj


def dataset_label_fix(dataset, fix_file):
    fix_data = pd.read_csv(fix_file, delimiter=",", encoding="utf-8")
    true_labels = dict()
    count = 0
    for index, row in fix_data.iterrows():
        true_labels[row['raw_id']] = row['label']
    for index in range(len(dataset.samples)):
        raw_id = dataset.samples[index][0]
        if raw_id in true_labels:
            dataset.labels[index] = true_labels[raw_id]
            count += 1
    print("%d records to be fixed, %d done." % (len(true_labels), count))
    return dataset


def compare_dataset(dataset, dataset_):
    id_set = set()
    for index in range(len(dataset.samples)):
        raw_id = dataset.samples[index][0]
        id_set.add(raw_id)
    hit_count = 0
    miss_count = 0
    for index in range(len(dataset_.samples)):
        raw_id = dataset_.samples[index][0]
        if raw_id in id_set:
            hit_count += 1
        else:
            miss_count += 1
    print("set size: %d, hit count: %d, miss count: %d" % (len(id_set), hit_count, miss_count))
    if miss_count == 0 and hit_count == len(id_set):
        return True
    else:
        return False


if __name__ == "__main__":
    dataset = pickle.load(open('../tmp/24581/unigram_test_dataset.pkl', 'rb'))
    dateset_ = pickle.load(open('../tmp/24581/triplet_test_dataset.pkl', 'rb'))
    compare_dataset(dataset, dateset_)
