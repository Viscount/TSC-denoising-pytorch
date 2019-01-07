#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np
import pandas as pd
import collections


class DmUnigramDataset(data.Dataset):
    def __init__(self, dm_samples, min_count, max_len, dictionary=None):
        self.max_len = max_len
        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')
            aggregate_sample = []
            for sample in dm_samples:
                aggregate_sample.extend(sample['content'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(aggregate_sample).most_common())
            rare_words = set()
            for word in counter:
                if word != 'UNK' and counter[word] <= min_count:
                    rare_words.add(word)
            for word in rare_words:
                counter['UNK'] += counter[word]
                counter.pop(word)
            print('%d words founded in vocabulary' % len(counter))

            self.vocab_counter = counter
            self.word_to_ix = {
                'EPT': 0
            }
            for word in counter:
                self.word_to_ix[word] = len(self.word_to_ix)

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = sample['label']
            sample_ = np.zeros(max_len, dtype=int)
            index = 0
            for word in sample['content']:
                sample_[index] = self.word2ix(word)
                index += 1
            self.samples.append((sample['raw_id'], np.array(sample_)))
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
            'label': label
        }
        return sample_dict

    def __len__(self):
        return len(self.samples)

    def save_vocab(self, path):
        vocab = []
        for word in self.vocab_counter:
            vocab.append({'idx': self.word_to_ix[word],
                          'word': word,
                          'count': self.vocab_counter[word]})
        df = pd.DataFrame(vocab)
        df.to_csv(path, index=False)
        return

    def vocab_size(self):
        return len(self.word_to_ix)