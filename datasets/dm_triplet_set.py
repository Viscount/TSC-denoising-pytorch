#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np
import pandas as pd
import collections
import random
import Levenshtein.StringMatcher


class DmTripletTrainDataset(data.Dataset):
    def __init__(self, dm_samples, min_count, max_len, context_size, max_distance, dictionary=None):
        self.max_len = max_len
        self.min_count = min_count
        self.max_distance = max_distance
        self.all_sentences = []
        self.sent_to_idx = dict()
        self.positive_samples = dict()

        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')
            aggregate_samples = []
            for episode_lvl_samples in dm_samples:
                for sample in episode_lvl_samples:
                    self.all_sentences.append(sample)
                    self.sent_to_idx[sample['raw_id']] = len(self.all_sentences)-1
                    aggregate_samples.extend(sample['content'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(aggregate_samples).most_common())
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

            print('building pinyin vocabulary...')
            py_aggregate_samples = []
            for episode_lvl_samples in dm_samples:
                for sample in episode_lvl_samples:
                    py_aggregate_samples.extend(sample['pinyin'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(py_aggregate_samples).most_common())
            rare_words = set()
            for word in counter:
                if word != 'UNK' and counter[word] <= min_count:
                    rare_words.add(word)
            for word in rare_words:
                counter['UNK'] += counter[word]
                counter.pop(word)
            print('%d pinyin words founded in vocabulary' % len(counter))

            self.py_vocab_counter = counter
            self.py_word_to_ix = {
                'EPT': 0
            }
            for word in counter:
                self.py_word_to_ix[word] = len(self.py_word_to_ix)

        print('building samples...')

        positive_count = 0
        for episode_lvl_samples in dm_samples:
            context_start_index = 0
            for sample in episode_lvl_samples:

                sample_id = sample['raw_id']
                playback_time = sample['playback_time']
                content = sample['content']
                py_content = sample['pinyin']

                # update context_start_index
                start_sample = episode_lvl_samples[context_start_index]
                while start_sample['playback_time'] < playback_time - context_size:
                    context_start_index += 1
                    start_sample = episode_lvl_samples[context_start_index]
                # find semantic similar comments in context
                it = context_start_index
                sample_ = episode_lvl_samples[it]
                sample_candidates = []
                while sample_['playback_time'] <= playback_time + context_size:
                    if sample['raw_id'] != sample_['raw_id']:
                            # and distance(py_content, sample_['pinyin']) <= max_distance:
                        sample_candidates.append(sample_['raw_id'])
                    it += 1
                    if it >= len(episode_lvl_samples):
                        break
                    else:
                        sample_ = episode_lvl_samples[it]

                if len(sample_candidates) > 0:
                    self.positive_samples[sample_id] = sample_candidates
                positive_count += len(sample_candidates)

        print('%d samples constructed.' % positive_count)
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def pyword2ix(self, word):
        if word in self.py_word_to_ix:
            return self.py_word_to_ix[word]
        else:
            return self.py_word_to_ix['UNK']

    def __getitem__(self, index):
        sample_id = list(self.positive_samples.keys())[index]
        sample = self.all_sentences[self.sent_to_idx[sample_id]]
        positive_id = random.choice(self.positive_samples[sample['raw_id']])
        positive = self.all_sentences[self.sent_to_idx[positive_id]]
        negative = negative_sampling(sample, self.all_sentences)

        sentence_anchor = tokenize(sample['content'], self.max_len, self.word2ix)
        sentence_positive = tokenize(positive['content'], self.max_len, self.word2ix)
        sentence_negative = tokenize(negative['content'], self.max_len, self.word2ix)
        py_anchor = tokenize(sample['pinyin'], self.max_len, self.pyword2ix)
        py_positive = tokenize(positive['pinyin'], self.max_len, self.pyword2ix)
        py_negative = tokenize(negative['pinyin'], self.max_len, self.pyword2ix)
        label = [sample['label'], positive['label'], negative['label']]

        mask = np.zeros(3, dtype=int)
        for i in range(3):
            if label[i] != -1:
                mask[i] = 1
        sample_dict = {
            'anchor': sentence_anchor,
            'pos': sentence_positive,
            'neg': sentence_negative,
            'py_anchor': py_anchor,
            'py_pos': py_positive,
            'py_neg': py_negative,
            'label': np.array([label[0], label[1], label[2]]),
            'mask': mask
        }
        return sample_dict

    def __len__(self):
        return len(list(self.positive_samples.keys()))

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

    def py_vocab_size(self):
        return len(self.py_word_to_ix)


class DmTripletTestDataset(data.Dataset):
    def __init__(self, dm_samples, max_len, word_dictionary, py_dictionary):
        self.max_len = max_len
        self.word_to_ix = word_dictionary
        self.py_word_to_ix = py_dictionary

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = sample['label']
            sample_ = tokenize(sample['content'], max_len, self.word2ix)
            py_sample_ = tokenize(sample['pinyin'], max_len, self.pyword2ix)
            self.samples.append((sample['raw_id'], sample_, py_sample_))
            self.labels.append(label)

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def pyword2ix(self, word):
        if word in self.py_word_to_ix:
            return self.py_word_to_ix[word]
        else:
            return self.py_word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        sample_dict = {
            'raw_id': sample[0],
            'sentence': sample[1],
            'pinyin': sample[2],
            'label': label
        }
        return sample_dict

    def __len__(self):
        return len(self.samples)


def common_words(content_a, content_b):
    word_set = set(content_a)
    count = 0
    for word in content_b:
        if word in word_set:
            count += 1
    return count


def distance(content_a, content_b):
    full_str_a = ''.join([word for word in content_a])
    full_str_b = ''.join([word for word in content_b])
    return Levenshtein.distance(full_str_a, full_str_b)


def tokenize(sentence, max_len, dictionary_func):
    result = np.zeros(max_len, dtype=int)
    index = 0
    for word in sentence:
        result[index] = dictionary_func(word)
        index += 1
    return result


def negative_sampling(sample, all_samples):
    content = sample['content']
    result = None
    while result is None:
        candidate = np.random.choice(len(all_samples), 1)
        sample_ = all_samples[candidate[0]]
        if common_words(content, sample_['content']) == 0:
            result = sample_
    return result
