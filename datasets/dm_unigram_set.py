#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np


class DmUnigramDataset(data.Dataset):
    def __init__(self, dm_samples, max_len, dictionary=None):
        self.max_len = max_len
        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = sample['label']
            sample_ = np.zeros(max_len, dtype=int)
            word_mask = np.zeros(max_len, dtype=int)
            index = 0
            for word in sample['content']:
                sample_[index] = self.word2ix(word)
                word_mask[index] = 1
                index += 1
            self.samples.append((sample['raw_id'], sample_, word_mask))
            self.labels.append(label)

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        sample_dict = {
            'raw_id': sample[0],
            'sentence': sample[1],
            'word_mask': sample[2],
            'label': label
        }
        return sample_dict

    def __len__(self):
        return len(self.samples)

    def vocab_size(self):
        return len(self.word_to_ix)
