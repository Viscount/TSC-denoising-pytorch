#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np
import pandas as pd
import collections


class DmTripletTrainDataset(data.Dataset):
    def __init__(self, dm_samples, min_count, max_len, context_size, min_common_words, dictionary=None):
        self.max_len = max_len
        self.min_count = min_count
        self.min_common_words = min_common_words
        all_sentences = []
        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')
            aggregate_samples = []
            for episode_lvl_samples in dm_samples:
                for sample in episode_lvl_samples:
                    all_sentences.append(sample)
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
            aggregate_samples = []
            for episode_lvl_samples in dm_samples:
                for sample in episode_lvl_samples:
                    all_sentences.append(sample)
                    aggregate_samples.extend(sample['pinyin'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(aggregate_samples).most_common())
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
        self.samples = []
        self.labels = []
        for episode_lvl_samples in dm_samples:
            context_start_index = 0
            for sample in episode_lvl_samples:

                playback_time = sample['playback_time']
                content = sample['content']
                py_content = sample['pinyin']

                # update context_start_index
                start_sample = episode_lvl_samples[context_start_index]
                while start_sample['playback_time'] < playback_time - context_size:
                    context_start_index += 1
                    start_sample = episode_lvl_samples[context_start_index]
                # build sample pair in context
                it = context_start_index
                sample_ = episode_lvl_samples[it]
                while sample_['playback_time'] <= playback_time + context_size:
                    if sample['raw_id'] != sample_['raw_id'] and \
                            common_words(py_content, sample_['pinyin']) >= min_common_words:
                        sentence_anchor = tokenize(sample['content'], max_len, self.word2ix)
                        sentence_positive = tokenize(sample_['content'], max_len, self.word2ix)
                        neg_sample = negative_sampling(sample, all_sentences)
                        sentence_negative = tokenize(neg_sample['content'], max_len, self.word2ix)
                        py_anchor = tokenize(sample['pinyin'], max_len, self.pyword2ix)
                        py_positive = tokenize(sample_['pinyin'], max_len, self.pyword2ix)
                        py_negative = tokenize(neg_sample['pinyin'], max_len, self.pyword2ix)
                        self.samples.append((sentence_anchor, sentence_positive, sentence_negative,
                                             py_anchor, py_positive, py_negative))
                        self.labels.append((sample['label'], sample_['label'], neg_sample['label']))
                    it += 1
                    if it >= len(episode_lvl_samples):
                        break
                    else:
                        sample_ = episode_lvl_samples[it]

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
        mask = np.zeros(3, dtype=int)
        for i in range(3):
            if label[i] != -1:
                mask[i] = 1
        sample_dict = {
            'anchor': sample[0],
            'pos': sample[1],
            'neg': sample[2],
            'py_anchor': sample[3],
            'py_pos': sample[4],
            'py_neg': sample[5],
            'label': np.array([label[0], label[1], label[2]]),
            'mask': mask
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
