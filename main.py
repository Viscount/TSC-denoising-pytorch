#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import util.dataset_util as dataset
import csv
import pickle
import random
import numpy as np
import torch
import os

import model.skip_gram as skip_gram
import model.tsc_embed as tsc_embed
import model.e2eshallow as e2e
import model.e2ewordembed as e2e_we
import model.e2ecnn as e2e_cnn
import model.e2epycnn as e2e_pycnn
import model.e2eselfattention as e2e_sa
import model.e2ernn as e2e_rnn
import model.supervised_rnn as sup_rnn
import model.supervised_cnn as sup_cnn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


def preprocess(season_id):
    # load data from csv
    danmaku_complete = pd.read_csv("./data/danmaku_complete.csv", delimiter="\t", encoding="utf-8",
                                   quoting=csv.QUOTE_NONE, low_memory=False)
    danmaku_complete = danmaku_complete.fillna(-1)

    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == season_id]

    samples, train_select, test_select = dataset.dataset_split(danmaku_selected)

    # pickle.dump(samples, open('./tmp/' + season_id + '_samples.pkl', 'wb'))
    # pickle.dump(train_select, open('./tmp/' + season_id + '_train_select.pkl', 'wb'))
    # pickle.dump(test_select, open('./tmp/' + season_id + '_test_select.pkl', 'wb'))

    # samples = pickle.load(open('./tmp/'+season_id+'_samples.pkl', 'rb'))
    # train_select = pickle.load(open('./tmp/'+season_id+'_train_select.pkl', 'rb'))
    # test_select = pickle.load(open('./tmp/'+season_id+'_test_select.pkl', 'rb'))
    # print(len(samples), len(train_select), len(test_select))

    dataset_type = 'unigram'

    dm_train_set, dm_test_set = dataset.build(samples, train_select, test_select, dataset_type)

    print('Dataset Exporting...')

    pickle.dump(dm_train_set, open(os.path.join('./tmp', season_id, dataset_type + '_train_dataset.pkl'), 'wb'))
    pickle.dump(dm_test_set, open(os.path.join('./tmp', season_id, dataset_type + '_test_dataset.pkl'), 'wb'))

    dataset_type = 'triplet'

    dm_train_set, dm_test_set = dataset.build(samples, train_select, test_select, dataset_type)

    print('Dataset Exporting...')

    pickle.dump(dm_train_set, open(os.path.join('./tmp', season_id, dataset_type + '_train_dataset.pkl'), 'wb'))
    pickle.dump(dm_test_set, open(os.path.join('./tmp', season_id, dataset_type + '_test_dataset.pkl'), 'wb'))

    # dataset_type = 'seperate'
    #
    # train_samples, test_samples, unlabeled_samples = dataset.build(samples, train_select, test_select, dataset_type)
    #
    # print('Dataset Exporting...')
    #
    # pickle.dump(train_samples, open('./tmp/train_samples.pkl', 'wb'))
    # pickle.dump(test_samples, open('./tmp/test_samples.pkl', 'wb'))
    # pickle.dump(unlabeled_samples, open('./tmp/unlabeled_samples.pkl', 'wb'))


if __name__ == "__main__":
    # set random seed
    set_random_seed(2333)
    season_id = '24581'
    # preprocess(season_id)

    # load dataset
    dataset_type = 'triplet'
    train_set = pickle.load(open(os.path.join('./tmp/', season_id, dataset_type+'_train_dataset.pkl'), 'rb'))
    test_set = pickle.load(open(os.path.join('./tmp/', season_id, dataset_type+'_test_dataset.pkl'), 'rb'))
    print(type(train_set))
    print(type(test_set))

    # test_set = dataset.dataset_label_fix(test_set, './data/label-fix.csv')

    # train
    # skip_gram.train(dataset)
    # tsc_embed.train(dataset)

    # models that fed with unigram dataset

    # e2e.train(season_id, train_set, test_set)
    # sup_rnn.train(season_id, train_set, test_set)
    # sup_cnn.train(season_id, train_set, test_set)

    # models that fed with triplet dataset

    # e2e_we.train(train_set, test_set)
    # e2e_cnn.train(train_set, test_set)
    # e2e_pycnn.train(season_id, train_set, test_set)
    # e2e_sa.train(train_set, test_set)
    # e2e_rnn.train(train_set, test_set)
