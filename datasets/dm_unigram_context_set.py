#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np


class DmUnigramContaxtDataset(data.Dataset):
    def __init__(self, dm_samples, select, max_len, context_size, context_words, dictionarys=None):
        self.max_len = max_len
        self.context_words = context_words
        self.context_size = context_size
        self.samples = []

        if dictionarys is not None:
            self.word_to_ix = dictionarys['vocab']
        else:
            print('No dictionaries')

        print('building samples...')

        for episode_lvl_samples in dm_samples:
            context_start_index = 0
            for sample in episode_lvl_samples:

                sample_id = sample['raw_id']
                if sample_id not in select:
                    continue
                playback_time = sample['playback_time']
                content = sample['content']
                py_content = sample['pinyin']

                # update context_start_index
                start_sample = episode_lvl_samples[context_start_index]
                while start_sample['playback_time'] < playback_time - context_size:
                    context_start_index += 1
                    start_sample = episode_lvl_samples[context_start_index]

                # find context
                it = context_start_index
                sample_ = episode_lvl_samples[it]
                context_samples = []
                while sample_['playback_time'] <= playback_time + context_size:
                    if sample['raw_id'] != sample_['raw_id']:
                        context_samples.append(sample_['content'])
                    it += 1
                    if it >= len(episode_lvl_samples):
                        break
                    else:
                        sample_ = episode_lvl_samples[it]

                content_ = np.zeros(max_len, dtype=int)
                index = 0
                for word in content:
                    content_[index] = self.word2ix(word)
                    index += 1

                context = np.zeros(context_words, dtype=int)
                context_count = np.zeros(context_words, dtype=int)
                context_word_count = dict()
                for sentence in context_samples:
                    for word in sentence:
                        word_token = self.word2ix(word)
                        if word_token in context_word_count:
                            context_word_count[word_token] += 1
                        else:
                            context_word_count[word_token] = 1
                count_pair = [(word_token, context_word_count[word_token]) for word_token in context_word_count]
                count_pair = sorted(count_pair, key=lambda x: x[1], reverse=True)
                for i in range(context_words):
                    if i >= len(count_pair):
                        break
                    context[i] = count_pair[i][0]
                    context_count[i] = count_pair[i][1]

                sample_dict = {
                    'raw_id': sample_id,
                    'sentence': content_,
                    'context': context,
                    'context_count': context_count,
                    'label': sample['label']
                }
                self.samples.append(sample_dict)

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample

    def __len__(self):
        return len(self.samples)

    def vocab_size(self):
        return len(self.word_to_ix)
